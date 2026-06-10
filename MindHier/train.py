import gc
import os
import sys
import time
import safetensors.torch
import torch
from trainer import MindHierTrainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
import wandb
import dist
from calculate_metrics import distributed_metrics_with_csv, to_PIL_image
from models import HierarchyAR, VQVAE, VQVAEHF, build_models
from models.basic_hierarchy_ar import AdaLNSelfCrossAttn
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.fsdp import load_model_state, save_model_state
from utils.lr_control import filter_params, lr_wd_annealing
from utils.data import build_dataset, coco_collate_fn
from utils.data_sampler import DistInfiniteBatchSampler
from utils.fid_score_in_memory import calculate_fid
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
from PIL import Image


def load_checkpoint_file(path: str):
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path)
    return torch.load(path, map_location="cpu")


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return checkpoint


def require_path(path: str, description: str, *, is_file: bool = False) -> None:
    if path is None:
        raise ValueError(f"{description} is required. Please set the corresponding command-line argument.")
    exists = os.path.isfile(path) if is_file else os.path.isdir(path)
    if not exists:
        expected = "file" if is_file else "directory"
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Expected a {expected}. This path is relative to the repository root unless an absolute path is provided."
        )


def validate_paths(args: arg_util.Args) -> None:
    require_path(args.data_path, "preprocessed NSD root")
    require_path(args.images_dir, "extracted NSD image directory")
    require_path(args.fmri_ckpt, "trained Stage 1 fMRI encoder checkpoint", is_file=True)
    require_path(args.vae_ckpt, "VQ-VAE/tokenizer checkpoint", is_file=True)
    if args.hierarchy_ar_ckpt is not None:
        require_path(args.hierarchy_ar_ckpt, "Stage 2 HierarchyAR checkpoint", is_file=True)
    os.makedirs(args.local_out_dir_path, exist_ok=True)


def build_everything(args: arg_util.Args):
    if args.multi_subject:
        raise ValueError(
            "Stage 2 currently expects a single subject per run because voxels are stacked into one tensor. "
            "Run one subject at a time, or refactor the dataloader/trainer to pass variable-length voxel lists."
        )
    validate_paths(args)
    # create tensorboard logger
    # tb_lg: misc.TensorboardLogger
    if dist.is_master():
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            dir=args.tb_log_dir_path,
            name=args.run_name if args.run_name is not None else 'test',
        )
        # os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.WandbLogger(),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)

    # log args
    print(f"initial args:\n{str(args)}")

    # build models
    vae_local, hierarchy_ar_wo_ddp, pipe = build_models(
        # VQVAE hyperparameters
        V=args.vqvae_vocab_size,
        Cvae=args.vqvae_channel_dim,
        ch=args.vqvae_n_channels,
        share_quant_resi=args.vqvae_share_quant_resi,
        # train hyperparameters
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        depth=args.depth,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        # text_encoder_path=args.text_encoder_path,
        # text_encoder_2_path=args.text_encoder_2_path,
        rope=args.rope,
        rope_theta=args.rope_theta,
        rope_size=args.rope_size,
        dpr=args.drop_path_rate,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        use_swiglu_ffn=args.use_swiglu_ffn,
        use_crop_cond=args.use_crop_cond,
        subj=args.subj,
        fmri_encoder_multi_subject=args.fmri_encoder_multi_subject,
    )
    # Load VAE and MindHier AR checkpoints
    if args.vae_ckpt is None or not os.path.exists(args.vae_ckpt):
        raise ValueError(
            "args.vae_ckpt is required. Set --vae_ckpt to your VQ-VAE checkpoint path."
        )
    vae_state_dict = unwrap_state_dict(load_checkpoint_file(args.vae_ckpt))
    vae_local.load_state_dict(vae_state_dict, strict=True)
    vae_local = vae_local.to(dist.get_device())
    if args.hierarchy_ar_ckpt is not None:
        if not os.path.exists(args.hierarchy_ar_ckpt):
            raise ValueError(
                "args.hierarchy_ar_ckpt does not exist. Set it to a valid Stage 2 checkpoint or omit it to train from scratch."
            )
        # Load the state dict
        state_dict = unwrap_state_dict(load_checkpoint_file(args.hierarchy_ar_ckpt))
        # Get current model's state dict shape
        model_state_dict = hierarchy_ar_wo_ddp.state_dict()
        # Filter to only keep parameters with matching shapes
        skipped_keys = [
            k for k, v in state_dict.items()
            if k not in model_state_dict or v.shape != model_state_dict[k].shape
        ]
        if skipped_keys:
            print(
                f"[WARN] Skipping {len(skipped_keys)} HierarchyAR checkpoint keys with missing or mismatched shapes: "
                f"{skipped_keys[:8]}"
            )
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        # Load the filtered state dict
        missing, unexpected = hierarchy_ar_wo_ddp.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"[WARN] HierarchyAR checkpoint missing {len(missing)} model keys after filtering: {missing[:8]}")
        if unexpected:
            print(f"[WARN] HierarchyAR checkpoint has {len(unexpected)} unexpected keys: {unexpected[:8]}")
    if args.fmri_ckpt is None or not os.path.exists(args.fmri_ckpt):
        raise ValueError(
            "args.fmri_ckpt is required. Set --fmri_ckpt to your trained Stage 1 checkpoint."
        )
    fmri_checkpoint = torch.load(args.fmri_ckpt, map_location="cpu")
    fmri_state_dict = unwrap_state_dict(fmri_checkpoint)
    missing, unexpected = pipe.fmri_vitl.load_state_dict(fmri_state_dict, strict=False)
    if missing:
        print(f"[WARN] Stage 1 checkpoint missing {len(missing)} keys for Stage 2 fMRI encoder: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Stage 1 checkpoint has {len(unexpected)} unexpected keys: {unexpected[:8]}")
    pipe.fmri_vitl.eval().requires_grad_(False)
    # start_it = load_model_state(args, hierarchy_ar_wo_ddp)
    start_it = 0
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hierarchy_ar_wo_ddp: HierarchyAR = args.compile_model(hierarchy_ar_wo_ddp, args.tfast)
    pipe.vae = vae_local
    pipe.hierarchy_ar = hierarchy_ar_wo_ddp
    pipe.switti = pipe.hierarchy_ar
    if args.use_gradient_checkpointing:
        hierarchy_ar_wo_ddp.enable_gradient_checkpointing()

    print(f"[INIT] MindHier HierarchyAR model = {hierarchy_ar_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters())/1e6:.2f}"
    print(f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}"
        for k, m in (
            ("VAE", vae_local),
            ("VAE.enc", vae_local.encoder),
            ("VAE.dec", vae_local.decoder),
            ("VAE.quant", vae_local.quantize),
    )]))
    print(
        f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("HierarchyAR", hierarchy_ar_wo_ddp),)])
        + "\n\n"
    )

    # FSDP wrapper
    hierarchy_ar: FSDP = (FSDP if (dist.initialized() and args.use_fsdp) else NullDDP)(
        hierarchy_ar_wo_ddp,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, AdaLNSelfCrossAttn),
        device_id=dist.get_local_rank(),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD if args.use_fsdp else ShardingStrategy.NO_SHARD, #FULL_SHARD,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    # build optimizer
    names, paras, para_groups = filter_params(hierarchy_ar, nowd_keys={
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    optimizer = torch.optim.AdamW(
        params=para_groups,
        lr=args.tlr, weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=args.afuse if not args.use_fsdp else False,
    )

    hierarchy_ar_optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # build data
    print(f"[build PT data] ...\n")
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")
    dataset_train = build_dataset(
        data_path=args.data_path,
        images_dir=args.images_dir,
        final_reso=args.data_load_reso,
        subj=args.subj,
        num_sessions=args.num_sessions,
        multi_subject=args.multi_subject,
        is_train=True,
        mid_reso=args.mid_reso,
    )
    
    ld_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        collate_fn=coco_collate_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train) if hasattr(dataset_train, '__len__') else 1000000,
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_it=0,
        ),
    )
    
    dataset_test = build_dataset(
        data_path=args.data_path,
        images_dir=args.images_dir,  # Need to provide this for NSDBrainDataset
        final_reso=args.data_load_reso,
        subj=args.subj,  # Need these additional parameters
        num_sessions=1,
        multi_subject=args.multi_subject,
        is_train=False
    )
    ld_test = DataLoader(
        dataset=dataset_test, num_workers=args.workers, pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        collate_fn=coco_collate_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_test), glb_batch_size=8*dist.get_world_size(), same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=False, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_it=start_it,
        ),
    )
    del dataset_test

    # build trainer
    trainer = MindHierTrainer(
        dataloader=ld_train,
        testloader=ld_test,
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        pipe=pipe,
        vae_local=vae_local,
        hierarchy_ar_wo_ddp=hierarchy_ar_wo_ddp,
        hierarchy_ar=hierarchy_ar,
        optimizer=hierarchy_ar_optimizer,
        label_smooth=args.ls,
        args=args,
    )
    torch.cuda.empty_cache()

    return (tb_lg, trainer, start_it)

def main_training():
    torch.set_num_threads(32)
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print('arg_init')
    (tb_lg, trainer, start_it) = build_everything(args)
    dist.barrier()

    # train
    
    for cur_iter in tqdm(range(start_it, args.max_iters)):
        tb_lg.set_step(cur_iter)

        # get current lr, wd
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.optimizer.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            cur_iter,
            args.wp,
            args.max_iters,
            wp0=args.wp0,
            wpe=args.wpe,
            wp_start_it=start_it,
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        # model forward-backward
        grad_norm, scale_log2 = trainer.train_step(g_it=cur_iter, tb_lg=tb_lg)
        
        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr, step=cur_iter)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr, step=cur_iter)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd, step=cur_iter)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd, step=cur_iter)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2, step=cur_iter)
        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm, step=cur_iter)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip, step=cur_iter)

        if cur_iter % args.save_iters == 0 and cur_iter > start_it:
            save_model_state(cur_iter, args, trainer.hierarchy_ar)
            
        if cur_iter > 0 and cur_iter % (args.log_iters) == 0:
            trainer.test_step(g_it=cur_iter, tb_lg=tb_lg)

    gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()

class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
