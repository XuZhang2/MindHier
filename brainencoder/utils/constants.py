"""Shared constants for Stage 1 fMRI-to-CLIP training and inference.

Readers adapting this repository should verify these values against their own
NSD preprocessing pipeline, especially the voxel counts and split sizes.
"""

from __future__ import annotations

from pathlib import Path


# Number of voxels after applying the NSD nsdgeneral mask for each subject.
# Replace these values if your preprocessing uses another ROI/mask.
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

NSD_NUM_IMAGES = 73000
NSD_TRIALS_PER_SESSION = 750
DEFAULT_TEST_SAMPLES = 3000

DEFAULT_DATA_ROOT = Path("data/nsd")
DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_OUTPUT_ROOT = Path("outputs/brainencoder")
DEFAULT_CLIP_CACHE_DIR = Path("pretrained/clip")
DEFAULT_CAPTIONS_PATH = Path("data/annotations/COCO_73k_annots_curated.npy")

CLIP_MODEL_IDS = {
    "VIT_L": "openai/clip-vit-large-patch14",
    "VIT_G": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
}

CLIP_FEATURE_DIMS = {
    "VIT_L": 768,
    "VIT_G": 1280,
}
