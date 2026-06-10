"""Model factory and public exports for MindHier Stage 2."""

from __future__ import annotations

import torch

from models.fmri_c2f_encoder import FMRI2CLIP
from models.hierarchy_ar import HierarchyAR, HierarchyARHF
from models.pipeline import MindHierPipeline
from models.vqvae import VQVAE, VQVAEHF


SUBJECT_NUM_VOXELS = {
    1: 15724,
    2: 14278,
    3: 15226,
    4: 13153,
    5: 13039,
    6: 17907,
    7: 12682,
    8: 14386,
}


def build_models(
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    device=None,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    depth=30,
    attn_l2_norm=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,
    rope=True,
    rope_theta=10000,
    rope_size=128,
    dpr=0.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    use_swiglu_ffn=True,
    use_crop_cond=False,
    subj=1,
    fmri_encoder_multi_subject=False,
):
    """Build the VQ-VAE, Stage 2 AR model, and inference pipeline."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
        test_mode=True,
    ).to(device)

    width = depth * 64
    hierarchy_ar = HierarchyAR(
        Cvae=Cvae,
        V=V,
        rope=rope,
        rope_theta=rope_theta,
        rope_size=rope_size,
        depth=depth,
        embed_dim=width,
        num_heads=depth,
        patch_nums=patch_nums,
        attn_l2_norm=attn_l2_norm,
        drop_path_rate=dpr,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        use_swiglu_ffn=use_swiglu_ffn,
        use_crop_cond=use_crop_cond,
    ).to(device)
    hierarchy_ar.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    fmri_vitl = FMRI2CLIP(
        input_dim=SUBJECT_NUM_VOXELS[subj],
        multi_subj=fmri_encoder_multi_subject,
    ).to(device)
    pipe = MindHierPipeline(hierarchy_ar, vae, fmri_vitl, device=device, dtype=torch.float32)
    return vae, hierarchy_ar, pipe


# Backward-compatible aliases.
Switti = HierarchyAR
SwittiHF = HierarchyARHF

__all__ = [
    "HierarchyAR",
    "HierarchyARHF",
    "MindHierPipeline",
    "Switti",
    "SwittiHF",
    "VQVAE",
    "VQVAEHF",
    "build_models",
]
