import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from models.vqvae import VQVAE, VQVAEHF
# from models.clip import FrozenCLIPEmbedder
from utils.clip import FrozenCLIPEmbedder
from models.hierarchy_ar import HierarchyARHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
import safetensors
import safetensors.torch

TRAIN_IMAGE_SIZE = (512, 512)


def load_checkpoint_file(path):
    if str(path).endswith(".safetensors"):
        return safetensors.torch.load_file(path)
    return torch.load(path, map_location='cpu')


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return checkpoint

class MindHierPipeline:
    vae_path = "yresearch/VQVAE-Switti"

    def __init__(self, hierarchy_ar, vae, fmri_vitl, 
                 device, dtype=torch.float32,
                 ):
        self.hierarchy_ar = hierarchy_ar.to(dtype)
        self.switti = self.hierarchy_ar  # Deprecated compatibility alias.
        self.vae = vae.to(dtype)
        self.fmri_vitl = fmri_vitl.to(dtype)
        self.clip_extractor = FrozenCLIPEmbedder("openai/clip-vit-large-patch14", device=device).to(dtype)
        # self.fmri_vitbig = fmri_vitbig.to(dtype)

        self.hierarchy_ar.eval()
        self.vae.eval()
        self.fmri_vitl.eval()
        # self.fmri_vitbig.eval()

        self.device = device

    def pretrained(self,
                        hierarchy_ar_path = None,
                        vae_ckpt = "pretrained/vqvae.safetensors",
                        fmri_vitl_ckpt = "pretrained/fmri_last.pth",
                        subj = 1,
                        torch_dtype=torch.bfloat16,
                        device=None,
                        reso=512,
                        ):
        if hierarchy_ar_path is None:
            raise ValueError(
                "Please pass hierarchy_ar_path with a valid Stage 2 checkpoint path."
            )
        hierarchy_ar_checkpoint = unwrap_state_dict(load_checkpoint_file(hierarchy_ar_path))
        self.hierarchy_ar.load_state_dict(hierarchy_ar_checkpoint, strict=True)
        self.vae.load_state_dict(unwrap_state_dict(load_checkpoint_file(vae_ckpt)), strict=True)
        checkpoint = torch.load(fmri_vitl_ckpt, map_location='cpu')
        self.fmri_vitl.load_state_dict(unwrap_state_dict(checkpoint), strict=False)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: torch.tensor ):
        with torch.no_grad():
            fmri_outputs = self.fmri_vitl(prompt)
            if len(fmri_outputs) == 3:
                fmri2img, _, fmri2img_all = fmri_outputs
            elif len(fmri_outputs) == 2:
                fmri2img, fmri2img_all = fmri_outputs
            else:
                raise ValueError(
                    "Stage 1 fMRI encoder must return either "
                    "(image_emb, text_emb, image_hierarchy) or (image_emb, image_hierarchy)."
                )
            fmri = fmri2img_all
            pooled_fmri2img_embeds = torch.mean(fmri2img, dim=1)

        return fmri, pooled_fmri2img_embeds, None

    def encode_prompt(
        self,
        prompt: torch.tensor ,
        null_prompt: torch.tensor = None,
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
        if encode_null:
            if null_prompt is None:
                null_prompt = torch.zeros_like(prompt[0]).unsqueeze(0)
            B, L, hidden_dim = prompt_embeds[0].shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)

            null_embeds = tuple(
                n_embed[:, :L].expand(B, L, hidden_dim).to(p_embed.device)
                for n_embed, p_embed in zip(null_embeds, prompt_embeds)
            )
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
            # null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            # prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            prompt_embeds = tuple(torch.cat([p_embed, n_embed], dim=0) 
                     for p_embed, n_embed in zip(prompt_embeds, null_embeds))
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            # attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: torch.tensor,
        null_prompt: torch.tensor = None,
        seed: int = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: float = None,
        recons_per_sample: int = 1,
        original_image: torch.tensor = None,
        clip_extractor = None,
        return_all: bool = False,
        verbose_timing: bool = True
    ):
        """
        Modified to support multiple reconstructions per sample EFFICIENTLY
        by batching the generation process.
        """
        # Initialize timing dictionary
        import time
        import torch
        timings = {}
        total_start = time.time()
        
        # Setup and initialization timing
        setup_start = time.time()
        assert not self.hierarchy_ar.training
        assert recons_per_sample > 0, "recons_per_sample must > 0"

        ar_model = self.hierarchy_ar
        vae = self.vae
        vae_quant = self.vae.quantize
        
        # [MODIFIED] Set seed once before the single generation pass
        if seed is not None:
            ar_model.rng.manual_seed(seed)
        rng = ar_model.rng if seed is not None else None

        # [MODIFIED] Get original batch size
        B_orig = prompt.shape[0]

        # [MODIFIED] Expand inputs to create a single large batch
        # This is the core change to avoid the for-loop
        prompt = prompt.repeat_interleave(recons_per_sample, dim=0)
        
        null_prompt = torch.zeros_like(prompt[0]).unsqueeze(0) # Keep null_prompt as size 1, it will be handled
        timings['setup'] = time.time() - setup_start
        
        # Prompt encoding timing
        encode_start = time.time()
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        timings['prompt_encoding'] = time.time() - encode_start
        
        # Conditioning preparation timing
        cond_prep_start = time.time()
        # [MODIFIED] The new effective batch size
        B = context[0].shape[0] // 2

        cond_vector = ar_model.fmri_pooler(cond_vector)

        if ar_model.use_crop_cond:
            crop_coords = get_crop_condition(
                2 * B * [TRAIN_IMAGE_SIZE[0]],
                2 * B * [TRAIN_IMAGE_SIZE[1]],
            ).to(cond_vector.device)
            crop_embed = ar_model.crop_embed(crop_coords.view(-1)).reshape(2 * B, ar_model.D)
            crop_cond = ar_model.crop_proj(crop_embed)
        else:
            crop_cond = None
        
        sos = cond_BD = cond_vector
        lvl_pos = ar_model.lvl_embed(ar_model.lvl_1L)
        if not ar_model.rope:
            lvl_pos += ar_model.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + ar_model.pos_start.expand(2 * B, ar_model.first_l, -1)
            + lvl_pos[:, : ar_model.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, ar_model.Cvae, ar_model.patch_nums[-1], ar_model.patch_nums[-1])
        timings['conditioning_prep'] = time.time() - cond_prep_start
        
        # KV cache clearing timing
        kv_clear_start = time.time()
        # Clear KV cache once before generation
        for b in ar_model.blocks:
            b.attn.kv_caching(ar_model.use_ar)
            b.cross_attn.kv_caching(True)
            if hasattr(b.attn, 'cached_k'):
                b.attn.cached_k = None
                b.attn.cached_v = None
            if hasattr(b.cross_attn, 'cached_k'):
                b.cross_attn.cached_k = None
                b.cross_attn.cached_v = None
        timings['kv_cache_clear'] = time.time() - kv_clear_start

        # [MODIFIED] The main `for recon_idx in range...` loop is REMOVED.
        # The following code now runs only ONCE on the expanded batch.
        
        # Main generation loop timing
        generation_start = time.time()
        stage_timings = []
        
        for si, pn in enumerate(ar_model.patch_nums):
            stage_start = time.time()
            
            ratio = si / ar_model.num_stages_minus_1
            x_BLC = next_token_map

            if ar_model.rope:
                freqs_cis = ar_model.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = ar_model.freqs_cis

            # CFG adjustment timing
            cfg_adjust_start = time.time()
            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = tuple(ctx[:B] for ctx in context)
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in ar_model.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth
            cfg_adjust_time = time.time() - cfg_adjust_start

            num_context = len(context)
            current_context = torch.cat(context[::-1], dim=1)
            
            # Transformer blocks timing
            blocks_start = time.time()
            for block_idx, block in enumerate(ar_model.blocks):
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=current_context,
                    num_context=num_context,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            blocks_time = time.time() - blocks_start
            
            cur_L += pn * pn

            # Logits computation timing
            logits_start = time.time()
            logits_BlV = ar_model.get_logits(x_BLC, cond_BD)
            logits_time = time.time() - logits_start
            
            # Guidance timing
            guidance_start = time.time()
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp
            guidance_time = time.time() - guidance_start

            # Sampling timing
            sampling_start = time.time()
            # [MODIFIED] Removed sampling diversity logic that depended on `recon_idx`.
            # The natural randomness of the sampling function on the larger batch is sufficient.
            current_top_k = top_k
            current_top_p = top_p
            
            if apply_smooth and si >= smooth_start_si:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=current_top_k, top_p=current_top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)
            sampling_time = time.time() - sampling_start

            # VAE quantization timing
            vae_quant_start = time.time()
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, ar_model.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                si, len(ar_model.patch_nums), f_hat, h_BChw,
            )
            if si != ar_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, ar_model.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    ar_model.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + ar_model.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.repeat(2, 1, 1)
            vae_quant_time = time.time() - vae_quant_start
            
            stage_time = time.time() - stage_start
            stage_timings.append({
                'stage': si,
                'total': stage_time,
                'cfg_adjust': cfg_adjust_time,
                'blocks': blocks_time,
                'logits': logits_time,
                'guidance': guidance_time,
                'sampling': sampling_time,
                'vae_quant': vae_quant_time
            })

        timings['generation_total'] = time.time() - generation_start
        timings['stage_details'] = stage_timings
        
        # Cleanup timing
        cleanup_start = time.time()
        for b in ar_model.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)
        timings['cleanup'] = time.time() - cleanup_start
        
        # VAE decoding timing
        vae_decode_start = time.time()
        # De-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        timings['vae_decode'] = time.time() - vae_decode_start
        
        # Reconstruction reshaping timing
        reshape_start = time.time()
        # [MODIFIED] Reshape the single large batch back into (recons, B_orig, C, H, W)
        # The subsequent selection logic can now work without changes.
        all_reconstructions = img.view(B_orig, recons_per_sample, *img.shape[1:]).transpose(0, 1)
        timings['reshape'] = time.time() - reshape_start
        
        # CLIP-based selection timing
        clip_selection_start = time.time()
        # Select best reconstruction if we have original image and CLIP extractor
        best_picks = None
        best_reconstruction = None
        import numpy as np
        import torch.nn as nn
        if original_image is not None and clip_extractor is not None and recons_per_sample > 1:
            B = all_reconstructions.shape[1]  # This is B_orig
            
            with torch.no_grad():
                # Target embeddings (already computed above)
                clip_image_target = clip_extractor.embed_image(original_image)
                clip_image_target_norm = nn.functional.normalize(
                    clip_image_target.flatten(1), dim=-1
                )  # Shape: [B, embed_dim]
                
                # Reshape all reconstructions into a single batch: [recons_per_sample * B, C, H, W]
                all_recons_flat = all_reconstructions.transpose(0, 1).reshape(
                    -1, *all_reconstructions.shape[2:]
                )
                
                # Single batched CLIP forward pass for ALL reconstructions
                all_recon_embeds = clip_extractor.embed_image(all_recons_flat.float())
                all_recon_embeds_norm = nn.functional.normalize(
                    all_recon_embeds.view(len(all_recon_embeds), -1), dim=-1
                )  # Shape: [recons_per_sample * B, embed_dim]
                
                # Reshape back to [recons_per_sample, B, embed_dim]
                all_recon_embeds_norm = all_recon_embeds_norm.view(recons_per_sample, B, -1)
                
                # Vectorized cosine similarity computation
                # clip_image_target_norm: [B, embed_dim]
                # all_recon_embeds_norm: [recons_per_sample, B, embed_dim]
                similarities = torch.cosine_similarity(
                    clip_image_target_norm.unsqueeze(0),  # [1, B, embed_dim]
                    all_recon_embeds_norm,                # [recons_per_sample, B, embed_dim]
                    dim=-1  # Compute similarity along embedding dimension
                )  # Result: [recons_per_sample, B]
                
                # Find best reconstruction for each sample
                best_picks = torch.argmax(similarities, dim=0).cpu().numpy()  # Shape: [B]
                
                # Gather the best reconstructions using advanced indexing
                sample_indices = torch.arange(B)
                best_reconstruction = all_reconstructions[best_picks, sample_indices]
        else:
            best_reconstruction = all_reconstructions[0]
            if recons_per_sample > 1:
                print(f"Generated {recons_per_sample} reconstructions. Returning first one (no selection criteria provided).")
        timings['clip_selection'] = time.time() - clip_selection_start
        
        # PIL conversion timing
        pil_convert_start = time.time()
        if return_pil:
            if best_reconstruction is not None:
                best_reconstruction = self.to_image(best_reconstruction)
            if return_all:
                all_reconstructions_pil = []
                for i in range(recons_per_sample):
                    all_reconstructions_pil.append(self.to_image(all_reconstructions[i]))
                all_reconstructions = all_reconstructions_pil
        timings['pil_convert'] = time.time() - pil_convert_start
        
        # Total timing
        timings['total'] = time.time() - total_start
        
        # Print timing summary if requested
        if verbose_timing:
            print("\n" + "="*60)
            print("TIMING BREAKDOWN")
            print("="*60)
            print(f"Total time: {timings['total']:.4f}s")
            print(f"Setup: {timings['setup']:.4f}s ({timings['setup']/timings['total']*100:.1f}%)")
            print(f"Prompt encoding: {timings['prompt_encoding']:.4f}s ({timings['prompt_encoding']/timings['total']*100:.1f}%)")
            print(f"Conditioning prep: {timings['conditioning_prep']:.4f}s ({timings['conditioning_prep']/timings['total']*100:.1f}%)")
            print(f"KV cache clear: {timings['kv_cache_clear']:.4f}s ({timings['kv_cache_clear']/timings['total']*100:.1f}%)")
            print(f"Generation total: {timings['generation_total']:.4f}s ({timings['generation_total']/timings['total']*100:.1f}%)")
            
            # Stage-by-stage breakdown
            print("\nPer-stage breakdown:")
            for stage_info in timings['stage_details']:
                si = stage_info['stage']
                print(f"  Stage {si}: {stage_info['total']:.4f}s")
                print(f"    - CFG adjust: {stage_info['cfg_adjust']:.4f}s")
                print(f"    - Blocks: {stage_info['blocks']:.4f}s")
                print(f"    - Logits: {stage_info['logits']:.4f}s")
                print(f"    - Guidance: {stage_info['guidance']:.4f}s")
                print(f"    - Sampling: {stage_info['sampling']:.4f}s")
                print(f"    - VAE quant: {stage_info['vae_quant']:.4f}s")
            
            print(f"\nCleanup: {timings['cleanup']:.4f}s ({timings['cleanup']/timings['total']*100:.1f}%)")
            print(f"VAE decode: {timings['vae_decode']:.4f}s ({timings['vae_decode']/timings['total']*100:.1f}%)")
            print(f"Reshape: {timings['reshape']:.4f}s ({timings['reshape']/timings['total']*100:.1f}%)")
            print(f"CLIP selection: {timings['clip_selection']:.4f}s ({timings['clip_selection']/timings['total']*100:.1f}%)")
            print(f"PIL convert: {timings['pil_convert']:.4f}s ({timings['pil_convert']/timings['total']*100:.1f}%)")
            print("="*60)
        
        # Store timings in the object for later access
        self.last_timings = timings
        
        if return_all:
            return best_reconstruction, all_reconstructions, best_picks
        else:
            return best_reconstruction


# Backward-compatible alias for old notebooks.
SwittiPipeline = MindHierPipeline
