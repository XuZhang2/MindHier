import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from pathlib import Path

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
                 clip_model_name_or_path="openai/clip-vit-large-patch14",
                 ):
        self.hierarchy_ar = hierarchy_ar.to(dtype)
        self.switti = self.hierarchy_ar  # Deprecated compatibility alias.
        self.vae = vae.to(dtype)
        self.fmri_vitl = fmri_vitl.to(dtype)
        self.clip_extractor = FrozenCLIPEmbedder(clip_model_name_or_path, device=device).to(dtype)
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
        prompt: torch.tensor ,
        null_prompt: torch.tensor = None,
        seed: int  = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp:  float = None
    ) -> torch.Tensor :
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        
        assert not self.hierarchy_ar.training
        ar_model = self.hierarchy_ar
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            ar_model.rng.manual_seed(seed)
            rng = ar_model.rng
        null_prompt = torch.zeros_like(prompt[0]).unsqueeze(0)
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

        B = context[0].shape[0] // 2

        cond_vector = ar_model.fmri_pooler(cond_vector)

        if ar_model.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
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

        for b in ar_model.blocks:
            b.attn.kv_caching(ar_model.use_ar)
            b.cross_attn.kv_caching(True)

        for si, pn in enumerate(ar_model.patch_nums):  # si: i-th segment
            ratio = si / ar_model.num_stages_minus_1
            x_BLC = next_token_map

            if ar_model.rope:
                freqs_cis = ar_model.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = ar_model.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                # context = context[:B]
                context = tuple(ctx[:B] for ctx in context)
                # context_attn_bias = context_attn_bias[:B]
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

            num_context = len(context)
            current_context = torch.cat(context[::-1], dim=1)
            # current_context = torch.cat(context, dim=1)
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
            cur_L += pn * pn

            logits_BlV = ar_model.get_logits(x_BLC, cond_BD)
            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, ar_model.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(ar_model.patch_nums), f_hat, h_BChw,
            )
            if si != ar_model.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, ar_model.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    ar_model.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + ar_model.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in ar_model.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)

        return img


# Backward-compatible alias for old notebooks.
SwittiPipeline = MindHierPipeline
