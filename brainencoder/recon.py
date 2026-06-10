import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime
from model.new_model import fMRI2CLIP

# from trainmodel import MindBridge, MindSingle
from utils.util import Clipper, reconstruction, read_responses_to_list
from utils.utils import seed_everything, get_dls
# from  utils import data
from utils.options import args
# from eval import cal_metrics
# from trainmodel.models import *
from utils.nsd_access import NSDAccess
from utils.constants import NSD_NUM_IMAGES, SUBJECT_NUM_VOXELS
import torch.nn as nn

## Load autoencoder
def prepare_voxel2sd(args, ckpt_path, device):
    try:
        from model.model import Voxel2StableDiffusionModel
    except ImportError as exc:
        raise ImportError(
            "The legacy low-level Stable Diffusion autoencoder is not included in this release. "
            "Set --img2img_strength 1 to disable img2img reconstruction, or add your local "
            "Voxel2StableDiffusionModel implementation before using this branch."
        ) from exc
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim=args.num_voxels)

    voxel2sd.load_state_dict(state_dict,strict=False)
    voxel2sd.to(device)
    voxel2sd.eval()
    print("Loaded low-level model!")

    return voxel2sd

def prepare_data(args):
    ## Load data
    args.num_voxels = SUBJECT_NUM_VOXELS[args.subj_test]

    # test_path = "{}/webdataset_avg_split/test/test_subj0{}".format(args.data_path, args.subj_test)
    # test_dl = data.get_dataloader(
    #     test_path,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     seed=args.seed,
    #     is_shuffle=False,
    #     extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy'],
    #     pool_type=args.pool_type,
    #     pool_num=args.pool_num,
    # )
    train_dl, val_dl, test_dl = get_dls(
    subject=args.subj_test,
    data_path=args.data_path,
    batch_size=args.batch_size,
    val_batch_size=1,
    num_workers=args.num_workers,
)

    return test_dl

def prepare_VD(args, device):
    print('Creating versatile diffusion reconstruction pipeline...')
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    try:
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(args.vd_cache_dir)
    except:
        print("Downloading Versatile Diffusion to", args.vd_cache_dir)
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
                "shi-labs/versatile-diffusion",
                cache_dir = args.vd_cache_dir)

    vd_pipe.image_unet.eval().to(device)
    vd_pipe.vae.eval().to(device)
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)

    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("shi-labs/versatile-diffusion", cache_dir=args.vd_cache_dir, subfolder="scheduler")

    # Set weighting of Dual-Guidance 
    # text_image_ratio=0.5 means equally weight text and image, 0 means use only image
    for name, module in vd_pipe.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = args.text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    return vd_pipe

def prepare_CLIP(args, device):
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes[args.clip_variant]
    out_dim_image = 257 * clip_size
    out_dim_text  = 77  * clip_size
    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)

    return clip_extractor, out_dim_image, out_dim_text

# def prepare_voxel2clip(args, out_dim_image, out_dim_text, device):
#     voxel2clip_kwargs = dict(
#         in_dim=args.pool_num, out_dim_image=out_dim_image, out_dim_text=out_dim_text, 
#         h=args.h_size, n_blocks=args.n_blocks, subj_list=args.subj_load)

#     # only need to load Single-subject version of MindBridge
#     voxel2clip = MindSingle(**voxel2clip_kwargs)

#     outdir = f'../train_logs/{args.model_name}'
#     ckpt_path = os.path.join(outdir, f'{args.ckpt_from}.pth')
#     print("ckpt_path",ckpt_path)
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     print("EPOCH: ",checkpoint['epoch'])
#     state_dict = checkpoint['model_state_dict']

#     voxel2clip.load_state_dict(state_dict,strict=False)
#     voxel2clip.requires_grad_(False)
#     voxel2clip.eval().to(device)

#     return voxel2clip

def prepare_voxel2embedding(device, rec_timestamp, train_logs_dir, subj_test):
    # voxel2emb = BrainXS(in_dim=15724, hidden_dim=1024, out_dim=768, num_latents=256)
    voxel2emb = fMRI2CLIP(input_dim=SUBJECT_NUM_VOXELS[subj_test])
    checkpoint_path = os.path.join(train_logs_dir, rec_timestamp, 'last.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    voxel2emb.load_state_dict(checkpoint['model_state_dict'])
    voxel2emb.to(device)
    voxel2emb.eval()
    print("Loaded voxel2embedding model!")
    return voxel2emb

def prepare_coco(args):
    # Preload coco captions
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, NSD_NUM_IMAGES))
    prompts_list = nsda.read_image_coco_info(coco_73k,info_type='captions')

    print("coco captions loaded.")

    return prompts_list

# def prepare_voxel2clip_MLP(args, out_dim_image, out_dim_text, device):

#     args.multi_voxel_dims = {1:15724, 2:14278, 5:13039, 7:12682}
#     model = MindEyeModule()
#     # model_g = MindEyeModule()
#     model.ridge = RidgeRegression(input_size=args.multi_voxel_dims[args.subj_test], out_features=2048)
#     model.backbone = BrainNetwork(in_dim=2048, latent_size=768, out_dim_image=out_dim_image, out_dim_text=out_dim_text, use_projector=True, train_type='vision')

#     # model_g.ridge = RidgeRegression(input_size=args.multi_voxel_dims[2], out_features=2048)
#     # model_g.backbone = BrainNetwork(in_dim=2048, latent_size=768, out_dim_image=out_dim_image, out_dim_text=out_dim_text, use_projector=True, train_type='vision') 
#     # import ipdb; ipdb.set_trace()vision_model_client1_softclip_avg_sim_best_600epoch_ema_text_20000mse_0727_200e_pretrain
#     # model_path_l = f'./logs/model/vision_model_client{args.subj_test}_softclip_sim_best_600epoch_ema_text_20000mse_0915.pth'
#     model_path_l = f'./logs/model/vision_model_client{args.subj_test}_softclip_round_100__text_20000mse_0927.pth'
#                                 # vision_model_client1_softclip_round_100__text_20000mse
#     # vision_model_client7_softclip_avg_sim_600_ema_text_20000mse_0727_200e_pretrain.pth


#     # model_path_g = f'./logs/model/global_model_softclip_avg_sim_best_600epoch_ema_text_20000mse.pth'
#     state_dict_l = torch.load(model_path_l, map_location=torch.device('cpu'))
#     # state_dict_g = torch.load(model_path_g, map_location=torch.device('cpu'))
#     # print(state_dict_l.keys())
#     model.load_state_dict(state_dict_l, strict=True)
#     # model_g.load_state_dict(state_dict_g, strict=True)
#     # import ipdb; ipdb.set_trace()
#     # load_filtered_state_dict(model.ridge, state_dict_l, 'ridge.')
#     # load_filtered_state_dict(model.backbone, state_dict_g, 'backbone.')

#     model.requires_grad_(False)
#     # model_g.requires_grad_(False)
#     model.eval().to(device)
#     # model_g.eval().to(device)
#     # return model, 
#     torch.cuda.empty_cache()
#     return model

def main(device):
    args.batch_size = 1
    if args.subj_load is None:
        args.subj_load = [args.subj_test]

    # Load data
    test_dl = prepare_data(args)
    num_test = len(test_dl)
    prompts_list = prepare_coco(args)
    # Load autoencoder
    outdir_ae = os.path.join(args.autoencoder_root, args.autoencoder_name, f'subj0{args.subj_test}')
    print(outdir_ae)
    ckpt_path = os.path.join(outdir_ae, f'epoch120.pth')
    if os.path.exists(ckpt_path):
        voxel2sd = prepare_voxel2sd(args, ckpt_path, device)
        # pool later
        args.pool_type = None
    else:
        print("No valid path for low-level model specified; not using img2img!") 
        args.img2img_strength = 1

    # Load VD pipeline
    vd_pipe = prepare_VD(args, device)
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    # Load CLIP
    clip_extractor, out_dim_image, out_dim_text = prepare_CLIP(args, device)

    # load voxel2clip
    voxel2clip = prepare_voxel2embedding(device, args.rec_timestamp, args.train_logs_dir, args.subj_test)

    outdir = os.path.join(args.recon_output_dir, args.model_name)
    save_dir = os.path.join(outdir, f"recon_on_subj{args.subj_test}")
    os.makedirs(save_dir, exist_ok=True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # define test range
    test_range = np.arange(num_test)
    if args.test_end is None:
        args.test_end = num_test
        
    # define recon logic
    only_lowlevel = False
    if args.img2img_strength == 1:
        img2img = False
    elif args.img2img_strength == 0:
        img2img = True
        only_lowlevel = True
    else:
        img2img = True

    clip_vision_train = None
    clip_text_train = None
    clip_vision_train_norm = None
    clip_text_train_norm = None

    if args.retrival_from_memory:
        # import numpy as np
        memory_path = os.path.join(args.memory_features_dir, f"subj0{args.subj_test}")
        # clip_vision_train = torch.load(memory_path + 'clipvision_train.pt')
        # clip_text_train = torch.load(memory_path + 'cliptext_train.pt')
        clip_vision_train = np.load(os.path.join(memory_path, 'nsd_clipvision_train.npy'))
        clip_text_train = np.load(os.path.join(memory_path, 'nsd_cliptext_train.npy'))
        clip_vision_train = torch.from_numpy(clip_vision_train).float()
        clip_text_train = torch.from_numpy(clip_text_train).float()
        clip_vision_train_norm = nn.functional.normalize(clip_vision_train.flatten(1), dim=-1)
        clip_text_train_norm = nn.functional.normalize(clip_text_train.flatten(1), dim=-1)
        # import ipdb; ipdb.set_trace()
    # embeddings_features = torch.zeros(982, 2048)
    # recon loop
    # response_list = read_responses_to_list(args.response_path)
    for val_i, (voxel, img, coco) in enumerate(tqdm(test_dl,total=len(test_range))):
        # import ipdb; ipdb.set_trace()
        if val_i < args.test_start:
            continue
        if val_i >= args.test_end:
            break
        if (args.samples is not None) and (val_i not in args.samples):
            continue

        voxel = torch.mean(voxel,axis=1).float().to(device)
        img = img.to(device)
        repeat_index = val_i % 3

        coco_squeezed = coco.squeeze()
        if coco_squeezed.dim() == 0:
            coco_ids = [coco_squeezed.item()]
        else:
            coco_ids = coco_squeezed.tolist()

        current_prompts_list = [prompts_list[coco_id] for coco_id in coco_ids]
        captions_gt = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]
        # import ipdb; ipdb.set_trace()
        # captions = response_list[val_i]
        # print(f'captioning: {captions}')
        print(f'GT_caption: {captions_gt}')
        

        with torch.no_grad():
            if args.only_embeddings:
                results = voxel2clip(voxel)
                embeddings = results[:2]
                torch.save(embeddings, os.path.join(save_dir, f'embeddings_{val_i}.pt'))
                continue
            if img2img: # will apply low-level and high-level pipeline
                ae_preds = voxel2sd(voxel)
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device)/0.18215).sample / 2 + 0.5

                # if val_i==0:
                #     plt.imshow(torch_to_Image(blurry_recons))
                #     plt.show()

                # pooling
                # voxel = data.pool_voxels(voxel, args.pool_num, args.pool_type)
            else: # only high-level pipeline
                blurry_recons = None

            if only_lowlevel: # only low-level pipeline
                brain_recons = blurry_recons
            else:
                grid, brain_recons, best_picks, recon_img = reconstruction(
                    args,
                    img, voxel, captions_gt, 
                    clip_vision_train, clip_text_train,
                    clip_vision_train_norm, clip_text_train_norm,
                    voxel2clip, clip_extractor, unet, vae, noise_scheduler,
                    img_lowlevel = blurry_recons,
                    num_inference_steps = args.num_inference_steps,
                    n_samples_save = args.batch_size,
                    recons_per_sample = args.recons_per_sample,
                    guidance_scale = args.guidance_scale,
                    img2img_strength = args.img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
                    seed = args.seed,
                    plotting = args.plotting,
                    verbose = args.verbose,
                    device = device,
                    mem_efficient = False,
                    retrival_from_memory = args.retrival_from_memory,
                    retrival_from_memory_strength = args.retrival_from_memory_strength,
                )

                if args.plotting:
                    grid.savefig(os.path.join(save_dir, f'{val_i}.png'))

                brain_recons = brain_recons[:,best_picks.astype(np.int8)]
                
                torch.save(img, os.path.join(save_dir, f'{val_i}_img.pt'))
                torch.save(brain_recons, os.path.join(save_dir, f'{val_i}_rec.pt'))


                # embeddings_features[val_i] = ridge_out.cpu()
                # import ipdb; ipdb.set_trace()
                # embeddings_features_np = embeddings_features.numpy()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("save path:", save_dir)

    # np.save(os.path.join(save_dir, f'embeddings_all.npy'), embeddings_features_np)
    # print("Saved embeddings_all.npy")

if __name__ == "__main__":
    seed_everything(seed=args.seed)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:",device)

    main(device)

    # args.results_path = f'../train_logs/{args.model_name}/recon_on_subj{args.subj_test}'
    # cal_metrics(args.results_path, device)
