import os
import argparse
from tqdm import tqdm
import time

import utils.utils as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
# from model.model import fMRI2CLIP
from model.new_model import fMRI2CLIP
# from model.fuse_txt_img_model import fMRI2CLIP
from utils.utils import soft_clip_loss
from utils.util import FrozenCLIPEmbedder
from utils.utils import prepare_coco
from utils.loss_avg import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.data import build_dataset, coco_collate_fn
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    CLIP_FEATURE_DIMS,
    CLIP_MODEL_IDS,
    DEFAULT_CAPTIONS_PATH,
    DEFAULT_CLIP_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_ROOT,
    NSD_TRIALS_PER_SESSION,
    SUBJECT_NUM_VOXELS,
)

torch.backends.cuda.matmul.allow_tf32 = True

from accelerate import Accelerator
accelerator = Accelerator(split_batches=False)
# accelerator = Accelerator(split_batches=False, mixed_precision='fp16')
print("PID of this process =", os.getpid())
print = accelerator.print

device = accelerator.device
print("device:", device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)


# configurations
parser = argparse.ArgumentParser(description='Model Training Configuration')
parser.add_argument(
    '--data_path',
    type=str,
    default=str(DEFAULT_DATA_ROOT),
    help='Preprocessed NSD root containing wds/ and beta HDF5 files.',
)
parser.add_argument(
    '--images_dir',
    type=str,
    default=str(DEFAULT_IMAGES_DIR),
    help='Directory containing image_000000.png ... image_072999.png.',
)
parser.add_argument(
    '--captions_path',
    type=str,
    default=str(DEFAULT_CAPTIONS_PATH),
    help='Path to COCO_73k_annots_curated.npy or an equivalent caption cache.',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default=str(DEFAULT_OUTPUT_ROOT),
    help='Directory for TensorBoard logs and checkpoints.',
)
parser.add_argument(
    '--clip_cache_dir',
    type=str,
    default=str(DEFAULT_CLIP_CACHE_DIR),
    help='Directory used by Hugging Face Transformers for CLIP weights.',
)
parser.add_argument(
    '--clip_local_files_only',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use only locally cached CLIP weights. Enable for offline servers.',
)
parser.add_argument('--vit_version', type=str, default='VIT_L', choices=['VIT_L', 'VIT_G'])
parser.add_argument('--subj', type=int, default=1, choices=list(SUBJECT_NUM_VOXELS.keys()))
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training')
parser.add_argument('--max_lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--mse_scale', type=float, default=200000.0)
parser.add_argument('--softclip_temp', type=float, default=0.005)
parser.add_argument('--final_reso', type=int, default=512)
parser.add_argument('--train_num_sessions', type=int, default=40)
parser.add_argument('--val_num_sessions', type=int, default=1)
parser.add_argument('--recon_loss', type=str, default='mse', choices=['mse', 'l1', 'huber', 'quantile'])
parser.add_argument('--use_image_aug', action=argparse.BooleanOptionalAction, default=False, help='whether to use image augmentation')
parser.add_argument(
    '--multi_subject',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Experimental legacy option. The paper pipeline uses single-subject Stage 1 training by default.',
)
parser.add_argument('--lr_scheduler_type', type=str, default='cycle', choices=['cycle','linear'])
args = parser.parse_args()

def require_path(path, description, *, is_file=False):
    if is_file:
        exists = os.path.isfile(path)
        expected = "file"
    else:
        exists = os.path.isdir(path)
        expected = "directory"
    if not exists:
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Expected a {expected}. This path is repository-relative by default; "
            "replace it with your local path if your data is stored elsewhere."
        )


def validate_paths(args):
    require_path(args.data_path, "preprocessed NSD root")
    require_path(args.images_dir, "extracted NSD image directory")
    require_path(args.captions_path, "COCO/NSD caption cache", is_file=True)
    if args.resume:
        require_path(args.resume, "Stage 1 resume checkpoint", is_file=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.clip_cache_dir, exist_ok=True)


validate_paths(args)
utils.seed_everything(args.seed, cudnn_deterministic=False)

run_name = (
    f"{'multi_subj' if args.multi_subject else 'subj'}_{args.vit_version.lower()}_"
    f"lr{args.max_lr:g}_layers12_16_20_24_subj{args.subj}"
)
log_dir = os.path.join(args.output_dir, run_name)
if local_rank == 0:
# tensorborad directory
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(log_dir, exist_ok=True)

print('\nprepare NSD webdataset data...')

print('\nprepare train and validation dataloaders...')

dataset_train = build_dataset(
    data_path=args.data_path,
    images_dir=args.images_dir,
    final_reso=args.final_reso,
    subj=args.subj,
    num_sessions=args.train_num_sessions,
    multi_subject=args.multi_subject,
    is_train=True,
    mid_reso=1,
)

train_dl = torch.utils.data.DataLoader(
    dataset=dataset_train,
    num_workers=num_workers,
    pin_memory=True,
    batch_size=args.batch_size,
    collate_fn=coco_collate_fn,
    shuffle=False, drop_last=True
)

dataset_test = build_dataset(
    data_path=args.data_path,
    images_dir=args.images_dir,
    final_reso=args.final_reso,
    subj=args.subj,
    num_sessions=args.val_num_sessions,
    multi_subject=False,
    is_train=False,
    mid_reso=1,
)

test_dl = torch.utils.data.DataLoader(
    dataset=dataset_test,
    num_workers=num_workers,
    pin_memory=True,
    batch_size=args.batch_size,
    collate_fn=coco_collate_fn,
    shuffle=False, drop_last=False
)

encoder_path = CLIP_MODEL_IDS[args.vit_version]
clip_extractor = FrozenCLIPEmbedder(
    encoder_path,
    device=device,
    cache_dir=args.clip_cache_dir,
    local_files_only=args.clip_local_files_only,
)

num_voxels = SUBJECT_NUM_VOXELS[args.subj]

voxel2emb = fMRI2CLIP(input_dim=num_voxels,
                        d_model=CLIP_FEATURE_DIMS[args.vit_version],
                        fmri_seq_len=100,
    image_seq_len=257,
    text_seq_len=77,
    num_layers=4,
    multi_subj=args.multi_subject)
print(voxel2emb)
# checkpoint = torch.load('train_logs/Feb21_21-15-20/last.pth', map_location='cpu')
# voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
voxel2emb.to(device)

print(voxel2emb)
if local_rank==0:
    utils.count_params(voxel2emb)

voxel2emb.requires_grad_(True)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2emb.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in voxel2emb.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)
num_samples_per_epoch = (NSD_TRIALS_PER_SESSION * args.train_num_sessions) // num_devices // args.batch_size

global_batch_size = args.batch_size * num_devices
if args.lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters = int(args.num_epochs*(num_samples_per_epoch)),
        last_epoch = -1
    )
elif args.lr_scheduler_type == 'cycle':
    total_steps = int(args.num_epochs*(num_samples_per_epoch)) # 240 * (8850 // 128)
    # one_epoch_steps = num_train // batch_size
    # one_epoch_steps = math.ceil(one_epoch_steps / num_devices)
    # total_steps = num_epochs * one_epoch_steps
    print(f"total_steps = {total_steps}")
    print(f"num_epochs = {args.num_epochs}")
    print(f"num_train = {num_samples_per_epoch}")
    print(f"global_batch_size = {global_batch_size}")
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr,
        total_steps=total_steps * num_devices,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/args.num_epochs
    )


def get_loss_func(recon_loss):
    loss_functions = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'quantile': lambda x, y: torch.quantile(torch.abs(x - y), 0.9)
    }
    if recon_loss not in loss_functions:
        raise ValueError(f"Unrecognized loss type: {recon_loss}")
    return loss_functions[recon_loss]



def save_ckpt(tag):    
    ckpt_path = log_dir + f'/{tag}.pth'
    print(f'\nsaving {ckpt_path}', flush=True)
    unwrapped_model = accelerator.unwrap_model(voxel2emb)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
            }, ckpt_path)
    except:
        print("Couldn't save... moving on to prevent crashing.")
    del unwrapped_model

print("\nDone with model preparations")


# main loop for training
if args.resume:
    print(f"loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    voxel2emb.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    lrs = checkpoint['lrs']
    print(f"resuming from epoch {epoch}")
    best_val_loss = min(val_losses)
else:
    # epoch = 0 if not resume else epoch + 1
    epoch = 0
    losses, val_losses, lrs = [], [], []
    best_val_loss = 1e9
    print(f"starting from scratch")

voxel2emb, optimizer, train_dl, test_dl, lr_scheduler = accelerator.prepare(
voxel2emb, optimizer, train_dl, test_dl,lr_scheduler
)

progress_bar = tqdm(range(epoch, args.num_epochs), ncols=120, disable=(local_rank!=0))

loss_fn = get_loss_func(args.recon_loss)

all_steps = 0

prompts_list = np.load(args.captions_path, allow_pickle=True)

total_loss = AverageMeter()
mse_image = AverageMeter()
mse_text = AverageMeter()
soft_clip_image = AverageMeter()
soft_clip_text = AverageMeter()
fwd_percent_correct_image = AverageMeter()
bwd_percent_correct_image = AverageMeter()
fwd_percent_correct_text = AverageMeter()
bwd_percent_correct_text = AverageMeter()
sim_image = AverageMeter()
sim_text = AverageMeter()
sim_image_former = [AverageMeter() for i in range(4)]
val_loss_base_mse_images_former = AverageMeter()
val_sims_image_former = [AverageMeter() for i in range(4)] # Assuming 4 intermediate layers

for epoch in progress_bar:
    voxel2emb.train()
    for train_i, (voxel, image, coco_id) in enumerate(train_dl):
        coco_id = coco_id.cpu().numpy()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            optimizer.zero_grad()
            repeat_index = train_i % 3
            # voxel = voxel[:,repeat_index].float()
            
            coco_ids = np.atleast_1d(coco_id.squeeze()).tolist()
            current_prompts_list = [prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index] for prompts in current_prompts_list]

            clip_image_pred, clip_text_pred, clip_images_pred_former = voxel2emb(voxel)
            clip_image, clip_images_former = clip_extractor.encode_image(image)
            clip_text = clip_extractor.encode_text(captions)
            clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
            clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)
            clip_image_former_pred_norm = [nn.functional.normalize(clip_images_pred_former[i].flatten(1), dim=-1) for i in range(len(clip_images_pred_former))]
            clip_image_former_norm = [nn.functional.normalize(clip_images_former[i].flatten(1), dim=-1) for i in range(len(clip_images_former))]


            clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
            clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)

            loss_mse_image = loss_fn(clip_image_pred_norm, clip_image_norm) * args.mse_scale
            loss_mse_text = loss_fn(clip_text_pred_norm, clip_text_norm) * args.mse_scale
            loss_mse_images_former = sum([loss_fn(clip_image_former_pred_norm[i], clip_image_former_norm[i]) for i in range(len(clip_images_former))]) * args.mse_scale

            loss_clip_image = soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                    temp = args.softclip_temp
                    )

            loss_clip_text = soft_clip_loss(
                    clip_text_pred_norm,
                    clip_text_norm,
                    temp = args.softclip_temp
                    )

            loss =  loss_mse_image + loss_mse_text + loss_clip_image + loss_clip_text + loss_mse_images_former

            utils.check_loss(loss)
            
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            sims_t = nn.functional.cosine_similarity(clip_text_norm, clip_text_pred_norm).mean().item()
            sims_img = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
            sims_img_former = [nn.functional.cosine_similarity(clip_image_former_norm[i], clip_image_former_pred_norm[i]).mean().item() for i in range(len(clip_images_former))]

            labels = torch.arange(len(clip_image_norm)).to(device) 
            fwd_percent_correct_img = utils.topk(utils.batchwise_cosine_similarity(clip_image_pred_norm,clip_image_norm), labels, k=1)
            bwd_percent_correct_img = utils.topk(utils.batchwise_cosine_similarity(clip_image_norm, clip_image_pred_norm), labels, k=1)

            labels = torch.arange(len(clip_text_norm)).to(device) 
            fwd_percent_correct_t = utils.topk(utils.batchwise_cosine_similarity(clip_text_pred_norm,clip_text_norm), labels, k=1)
            bwd_percent_correct_t = utils.topk(utils.batchwise_cosine_similarity(clip_text_norm, clip_text_pred_norm), labels, k=1)

            if args.lr_scheduler_type is not None:
                lr_scheduler.step()

            total_loss.update(loss.item())
            mse_image.update(loss_mse_image.item())
            mse_text.update(loss_mse_text.item())
            soft_clip_image.update(loss_clip_image.item())
            soft_clip_text.update(loss_clip_text.item())
            fwd_percent_correct_image.update(fwd_percent_correct_img)
            bwd_percent_correct_image.update(bwd_percent_correct_img)
            fwd_percent_correct_text.update(fwd_percent_correct_t)
            bwd_percent_correct_text.update(bwd_percent_correct_t)
            sim_image.update(sims_img)
            for i in range(len(clip_images_former)):
                sim_image_former[i].update(sims_img_former[i])
            sim_text.update(sims_t)

            if local_rank == 0:
                writer.add_scalar(f'Loss/loss_all', total_loss.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'Loss_mse/image', mse_image.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'Loss_mse/text', mse_text.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'Loss_soft_clip/image', soft_clip_image.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'Loss_soft_clip/text', soft_clip_text.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'percent_correct_fwd/image', fwd_percent_correct_image.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'percent_correct_bwd/image', bwd_percent_correct_image.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'percent_correct_fwd/text', fwd_percent_correct_text.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'percent_correct_bwd/text', bwd_percent_correct_text.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'sim/image', sim_image.avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'sim/text', sim_text.avg, epoch * len(train_dl) + train_i)
                for i in range(len(sim_image_former)):
                    writer.add_scalar(f'sim/image_former{i}', sim_image_former[i].avg, epoch * len(train_dl) + train_i)
                writer.add_scalar(f'lrs/lr', optimizer.param_groups[0]['lr'], epoch * len(train_dl) + train_i)

    all_steps += 1

    voxel2emb.eval()
    with torch.no_grad():
        val_sims_base_image = AverageMeter()
        val_sims_base_text = AverageMeter()
        val_loss_base_mse_image = AverageMeter()
        val_loss_base_mse_text = AverageMeter()
        val_loss_base_nce_image = AverageMeter()
        val_loss_base_nce_text = AverageMeter()

        for val_i, (voxel, image, coco_id) in enumerate(test_dl):

            repeat_index = val_i % 3
            # voxel = torch.mean(voxel, axis=1).float()
            coco_ids = np.atleast_1d(coco_id.squeeze()).tolist()
            current_prompts_list = [prompts_list[coco_id] for coco_id in coco_ids]
            captions = [prompts[repeat_index] for prompts in current_prompts_list]

            clip_image, clip_images_former = clip_extractor.encode_image(image)
            clip_text = clip_extractor.encode_text(captions)

            # Get predicted embeddings for final and intermediate layers
            clip_image_pred, clip_text_pred, clip_images_pred_former = voxel2emb(voxel)

            clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
            clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)
            # Normalize intermediate image embeddings
            clip_image_former_pred_norm = [nn.functional.normalize(clip_images_pred_former[i].flatten(1), dim=-1) for i in range(len(clip_images_pred_former))]
            clip_image_former_norm = [nn.functional.normalize(clip_images_former[i].flatten(1), dim=-1) for i in range(len(clip_images_former))]

            clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
            clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)    

            val_loss_mse_image = loss_fn(clip_image_pred_norm, clip_image_norm) * args.mse_scale
            val_loss_mse_images_former = sum([loss_fn(clip_image_former_pred_norm[i], clip_image_former_norm[i]) for i in range(len(clip_images_former))]) * args.mse_scale
            val_loss_mse_text = loss_fn(clip_text_pred_norm, clip_text_norm) * args.mse_scale

            loss_clip_image = soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                    temp = args.softclip_temp
                    )

            loss_clip_text = soft_clip_loss(
                    clip_text_pred_norm,
                    clip_text_norm,
                    temp = args.softclip_temp
                    )       

            val_sims_image = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
            val_sims_img_former = [nn.functional.cosine_similarity(clip_image_former_norm[i], clip_image_former_pred_norm[i]).mean().item() for i in range(len(clip_images_former))]
            val_sims_text = nn.functional.cosine_similarity(clip_text_norm, clip_text_pred_norm).mean().item()

            val_loss_base_nce_image.update(loss_clip_image.item())
            val_loss_base_nce_text.update(loss_clip_text.item())
            val_loss_base_mse_image.update(val_loss_mse_image.item())
            val_loss_base_mse_text.update(val_loss_mse_text.item())
            val_sims_base_image.update(val_sims_image)
            val_sims_base_text.update(val_sims_text)
            val_loss_base_mse_images_former.update(val_loss_mse_images_former.item())
            for i in range(len(clip_images_former)):
                val_sims_image_former[i].update(val_sims_img_former[i])

    torch.cuda.empty_cache() 

    if local_rank == 0:
        writer.add_scalar(f'Val/sim_image', val_sims_base_image.avg, all_steps)
        writer.add_scalar(f'Val/sim_text', val_sims_base_text.avg, all_steps)
        writer.add_scalar(f'Val/loss_mse_image', val_loss_base_mse_image.avg, all_steps)
        writer.add_scalar(f'Val/loss_mse_text', val_loss_base_mse_text.avg, all_steps)
        writer.add_scalar(f'Val/loss_SoftClip_image', val_loss_base_nce_image.avg, all_steps)
        writer.add_scalar(f'Val/loss_SoftClip_text', val_loss_base_nce_text.avg, all_steps)
        writer.add_scalar(f'Val/loss_mse_images_former', val_loss_base_mse_images_former.avg, all_steps)
        for i in range(len(val_sims_image_former)):
            writer.add_scalar(f'Val/sim/image_former_{i}', val_sims_image_former[i].avg, all_steps)
           
    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()

# training and validation loop ends here
# draw and save plots of training and validation losses
save_ckpt(f'last-hierarchy-vitl')
