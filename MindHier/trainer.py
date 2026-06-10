import math
from typing import List, Optional, Tuple, Union, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

import dist # Assuming this is your distributed training utilities
# Assuming these are your project's modules, adjust paths as necessary
from models import HierarchyAR, VQVAE
from models.pipeline import MindHierPipeline
from utils.amp_sc import AmpOptimizer
from utils.misc import TensorboardLogger
from utils.clip_score import FrozenCLIPEmbedder, CLIPScoreCalculator, ScoreTextImageOverlay

from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class MindHierTrainer(object):
    def __init__(
        self,
        dataloader, # Train dataloader
        testloader, # Test dataloader
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        pipe: MindHierPipeline,
        vae_local: VQVAE,
        hierarchy_ar_wo_ddp: HierarchyAR,
        hierarchy_ar: Union[DDP, FSDP], # Model can be DDP or FSDP wrapped
        optimizer: AmpOptimizer,
        label_smooth: float,
        args=None, # Namespace object for various arguments
    ):
        super().__init__()
        self.dataloader = iter(dataloader)
        if testloader is not None:
            self.testloader_iter: Optional[Iterator] = iter(testloader)
        else:
            self.testloader_iter = None
        self.args = args

        self.hierarchy_ar, self.vae_local, self.quantize_local = (
            hierarchy_ar,
            vae_local,
            vae_local.quantize,
        )
        self.switti = self.hierarchy_ar  # Backward-compatible alias.
        self.hierarchy_ar_wo_ddp: HierarchyAR = hierarchy_ar_wo_ddp
        self.switti_wo_ddp = self.hierarchy_ar_wo_ddp  # Backward-compatible alias.
        self.optimizer = optimizer
        self.pipe = pipe
        encoder_path = "openai/clip-vit-large-patch14"
        # encoder_path2 = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

        clip_extractor = FrozenCLIPEmbedder(encoder_path, device=device)
        self.clip_calculator = CLIPScoreCalculator(clip_extractor)
        # Assign RNG to the unwrapped model if it's an attribute
        if hasattr(self.hierarchy_ar_wo_ddp, 'rng'):
             self.hierarchy_ar_wo_ddp.rng = torch.Generator(device=device)


        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        # Ensure loss_weight is created on the correct device
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.device = device
        self.grad_accum = args.grad_accum
        self.embed_noise_std = args.embed_noise_std
        
    @torch.inference_mode()
    def _log_metrics_and_images(
        self,
        g_it: int,
        tb_lg: TensorboardLogger,
        data_source_prefix: str,
        image_batch: Ten,
        prompt_batch_tokens: Ten, # Expects tokenized prompts as a Tensor
    ):

        # 1. Prepare inputs for metric calculation
        inp_B3HW = image_batch.to(self.device, non_blocking=True)
        inp_B3HW = F.interpolate(
            inp_B3HW, size=(self.resos[-1], self.resos[-1]), mode="bicubic",
        )
        B, V = inp_B3HW.size(0), self.vae_local.vocab_size

        # For metrics (acc, CE), use ground truth from VAE without reconstruction noise
        gt_idx_Bl_eval: List[ITen] = self.vae_local.img_to_idxBl(
            inp_B3HW, noise_std=0.0 # No noise for consistent evaluation
        )
        gt_BL_eval = torch.cat(gt_idx_Bl_eval, dim=1)
        
        # Teacher-forcing input for the hierarchy autoregressive model.
        x_BLCv_wo_first_l_eval: Ten = self.quantize_local.idxBl_to_hierarchy_ar_input(gt_idx_Bl_eval)

        (prompt_embeds,
         pooled_prompt_embeds,
         prompt_attn_bias) = self.pipe.encode_prompt(
            prompt_batch_tokens.to(self.device, non_blocking=True),
            encode_null=False
        )

        with torch.no_grad(), self.optimizer.amp_ctx:
            logits_BLV = self.hierarchy_ar(
                x_BLCv_wo_first_l_eval,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_attn_bias=prompt_attn_bias,
            )

        # Compute cluster usage
        pred_BL = logits_BLV.data.argmax(dim=-1)
        # Ensure V is positive before division and bincount minlength
        cluster_usage = 0.0
        
        prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float().to(self.device) # Ensure correct device
        dist.allreduce(prob_per_class_is_chosen)
        prob_per_class_is_chosen /= (prob_per_class_is_chosen.sum() + 1e-8) # Add epsilon for stability
        cluster_usage = (prob_per_class_is_chosen > (0.001 / V)).float().mean().item() * 100


        logits_lg_dict = {}
        kw_dict = {
            f"{data_source_prefix}_z_voc_usage": cluster_usage,
            f"{data_source_prefix}_acc_total": 0.0,
            f"{data_source_prefix}_L_total": 0.0
        }

        for si, (bg, ed) in enumerate(self.begin_ends):
            # Ensure V is positive for reshaping and topk
            acc, acc_top5, ce, std, norm = 0.0, 0.0, 0.0, 0.0, 0.0
            pred_slice = logits_BLV.data[:, bg:ed]

            pred = pred_slice.reshape(-1, V)
            tar = gt_BL_eval[:, bg:ed].reshape(-1)
            
            top5_k = min(5, V) # k for top_k cannot exceed V
            top5_indices = torch.topk(pred, top5_k, dim=-1)[1]

            acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
            acc_top5 = torch.eq(tar[:, None], top5_indices).any(dim=1).float().mean().item() * 100
            ce = self.val_loss(pred, tar).item()
            std = pred.std(dim=-1).mean().item()
            norm = pred.norm(dim=-1).mean().item()

            stats = torch.tensor([acc, acc_top5, ce, std, norm], device=dist.get_device())
            dist.allreduce(stats)
            stats /= dist.get_world_size()
            acc, acc_top5, ce, std, norm = stats.tolist()

            logits_lg_dict[f"{data_source_prefix}_logits_std_{self.resos[si]}"] = std
            logits_lg_dict[f"{data_source_prefix}_logits_norm_{self.resos[si]}"] = norm
            kw_dict[f"{data_source_prefix}_acc_{self.resos[si]}"] = acc
            kw_dict[f"{data_source_prefix}_acc_top5_{self.resos[si]}"] = acc_top5
            kw_dict[f"{data_source_prefix}_L_{self.resos[si]}"] = ce
            if len(self.begin_ends) > 0:
                kw_dict[f"{data_source_prefix}_acc_total"] += acc / len(self.begin_ends)
                kw_dict[f"{data_source_prefix}_L_total"] += ce / len(self.begin_ends)

        # Image logging part
        if g_it % self.args.log_images_iters == 0 and B > 0: # Check B > 0 for image logging
            
            def generate_and_log_images_internal():
                torch.cuda.empty_cache()
                num_samples_to_log = min(B, 8)
                if num_samples_to_log == 0: return

                for cfg_val in [0, 6]:
                    # Generate images -> Immediately move to CPU
                    current_prompts_for_sampling = prompt_batch_tokens[:num_samples_to_log].to(self.device)
                    imgs_generated = self.pipe(current_prompts_for_sampling,
                                     cfg=cfg_val,
                                     top_k=self.args.top_k,
                                     top_p=self.args.top_p,
                                     return_pil=False,
                                     turn_on_cfg_start_si=1
                                     ).cpu()  # Original generation code
                    
                    # ==== Start Modification ====
                    imgs_generated_cpu = imgs_generated.cpu()  # Move to CPU immediately
                    del imgs_generated  # Delete GPU tensor
                    torch.cuda.empty_cache()
                    
                    # Process on CPU
                    imgs_transformed = imgs_generated_cpu.add(imgs_generated_cpu).add_(-1)
                    image_to_display = image_batch[:num_samples_to_log].cpu()  # Keep on CPU
                    clip_scores_tensor = self.clip_calculator(F.interpolate(imgs_transformed, size=(224, 224), mode='bicubic', align_corners=False).to(self.device), \
                                                            F.interpolate(image_to_display, size=(224, 224), mode='bicubic', align_corners=False).to(self.device))                  
                    concatenated_imgs = torch.cat((imgs_transformed, image_to_display), dim=2)
                                      
                    # Final grid creation
                    concatenated_imgs_01 = (concatenated_imgs + 1) / 2
                    
                    score_overlay = ScoreTextImageOverlay()
                    concatenated_imgs_01 = score_overlay.forward(concatenated_imgs_01, clip_scores_tensor)
                    
                    grid = make_grid(concatenated_imgs_01, nrow=num_samples_to_log, padding=0, pad_value=1.0)
                    tb_lg.log_image(
                        f"{data_source_prefix}_imgs_cfg{cfg_val}_top_k={self.args.top_k}_top_p={self.args.top_p}",
                        grid,
                        step=g_it,
                    )
                    del imgs_transformed, image_to_display
                    del concatenated_imgs, concatenated_imgs_01, grid
                    torch.cuda.empty_cache()
            
            is_fsdp = isinstance(self.hierarchy_ar, FSDP)
            if is_fsdp:
                # Summon full params for the FSDP wrapped model before calling the pipe
                with FSDP.summon_full_params(self.hierarchy_ar, writeback=False):
                    generate_and_log_images_internal()
            else:
                generate_and_log_images_internal()

        if dist.is_master():
            tb_lg.update(head=f"{data_source_prefix}_Logits_stats", **logits_lg_dict, step=g_it)
            tb_lg.update(head=f"{data_source_prefix}_AR_iter_loss", **kw_dict, step=g_it)


    def train_step(
        self,
        g_it: int,
        tb_lg: TensorboardLogger
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.hierarchy_ar.train()
        grad_norm_total, scale_log2_total = 0.0, 0.0
        last_image_for_log, last_prompt_tokens_for_log = None, None

        # --- Modified gradient accumulation loop ---
        for accum_iter in range(self.grad_accum):
            try:
                image, prompt_tokens = next(self.dataloader)
            except StopIteration:
                if accum_iter == 0:
                    return None, None
                break

            last_image_for_log = image.cpu()
            last_prompt_tokens_for_log = prompt_tokens

            # Forward pass
            inp_B3HW = image.to(self.device, non_blocking=True)
            inp_B3HW = F.interpolate(
                inp_B3HW, size=(self.resos[-1], self.resos[-1]), mode="bicubic"
            )

            B, V = inp_B3HW.size(0), self.vae_local.vocab_size

            # Compute loss
            gt_idx_Bl = self.vae_local.img_to_idxBl(inp_B3HW, noise_std=self.embed_noise_std)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_hierarchy_ar_input(gt_idx_Bl)

            if self.args.uncond_proba > 0:
                cond_uncond_choice = torch.bernoulli(
                    torch.full((B, ), self.args.uncond_proba)
                )
                for i_, p_ in enumerate(cond_uncond_choice):
                    if p_ == 1:
                        prompt_tokens[i_] = torch.zeros_like(prompt_tokens[i_])
            prompt_tokens = prompt_tokens.to(self.device, non_blocking=True)
            # Forward through model
            (prompt_embeds, pooled_prompt_embeds, prompt_attn_bias) = self.pipe.encode_prompt(prompt_tokens, encode_null=False)
            with self.optimizer.amp_ctx:
                logits_BLV = self.hierarchy_ar(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                )
                loss = self.train_loss(logits_BLV.view(-1, V),
                                       gt_BL.view(-1),
                                       ).view(B, -1)
                loss = loss.mul(self.loss_weight).sum(dim=-1).mean()

            # Backward and clip gradients (step only on last microbatch)
            # backward
            is_stepping = (accum_iter + 1) == self.grad_accum
            grad_norm, scale_log2 = self.optimizer.backward_clip_step(
                loss=loss,
                is_stepping=is_stepping,
                )

        # Log to tensorboard
        if g_it > 0 and g_it % self.args.log_iters == 0:
            self.hierarchy_ar.eval()
            
            if self.args.use_gradient_checkpointing:
                self.hierarchy_ar.disable_gradient_checkpointing()

            # Log training metrics using the last successfully processed batch from training
            if last_image_for_log is not None and last_prompt_tokens_for_log is not None:
                 self._log_metrics_and_images(
                    g_it,
                    tb_lg,
                    "train", # Prefix for training logs
                    last_image_for_log,
                    last_prompt_tokens_for_log 
                )
            self.hierarchy_ar.train()
            if self.args.use_gradient_checkpointing:
                self.hierarchy_ar.enable_gradient_checkpointing()
        
            dist.barrier() # Synchronize all processes after logging
            torch.cuda.empty_cache()

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, \
               scale_log2.item() if isinstance(scale_log2, torch.Tensor) else scale_log2
               
    @torch.inference_mode()
    def test_step(self,
        g_it: int,
        tb_lg: TensorboardLogger):
                    # Log test metrics
        self.hierarchy_ar.eval()
        if self.args.use_gradient_checkpointing:
            self.hierarchy_ar.disable_gradient_checkpointing()

        test_image_batch, test_prompt_batch_tokens = next(self.testloader_iter)
        self._log_metrics_and_images(
            g_it,
            tb_lg,
            "test", # Prefix for test logs
            test_image_batch,
            test_prompt_batch_tokens
        )

        self.hierarchy_ar.train()
        if self.args.use_gradient_checkpointing:
            self.hierarchy_ar.enable_gradient_checkpointing()
        if dist.is_master():
            print(f"LOGGING {g_it} FINISHED")
        dist.barrier() # Synchronize all processes after logging
        torch.cuda.empty_cache()
        
    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }


# Backward-compatible alias for older scripts.
SwittiTrainer = MindHierTrainer
