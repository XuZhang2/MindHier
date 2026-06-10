# MindHier

[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](#citation)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.22335-b31b1b.svg)](https://arxiv.org/abs/2510.22335)

Official code for **Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction**.

MindHier reconstructs images from fMRI in two stages:

1. **Stage 1 (`brainencoder/`)** trains a hierarchical fMRI encoder aligned to CLIP ViT-L/14 image and text features.
2. **Stage 2 (`MindHier/`)** trains a scale-aware autoregressive image generator conditioned on the Stage 1 hierarchy.

The repository keeps the original two-folder research layout so the released code is easy to audit.

## What You Need to Run

The actual training entry points are:

```text
brainencoder/train.py     # Stage 1 training
MindHier/train.py         # Stage 2 training
```

These scripts read command-line arguments directly. They do **not** currently load `configs/default.yaml`.

`configs/default.yaml` is only a human-readable reference config that summarizes common paths and hyperparameters. It is included for documentation and future cleanup, not as a required runtime file.

## Repository Layout

```text
.
├── brainencoder/          # Stage 1: fMRI -> hierarchical CLIP features
│   ├── train.py
│   ├── recon.py           # legacy reconstruction helper
│   ├── eval.py
│   ├── model/
│   └── utils/
├── MindHier/              # Stage 2: hierarchy-conditioned AR generator
│   ├── train.py
│   ├── trainer.py
│   ├── calculate_metrics.py
│   ├── models/
│   └── utils/
├── configs/default.yaml   # reference only; not loaded by current scripts
└── requirements.txt
```

Some notebooks, shell scripts, legacy model variants, and duplicated helper files are kept for transparency with the original research code. The minimal paper reproduction path is the two training commands shown below.

## Installation

Using conda:

```bash
conda create -n mindhier python=3.11
conda activate mindhier
pip install -r requirements.txt
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default one is not suitable for your machine.

## Data Preparation

MindHier uses the **Natural Scenes Dataset (NSD)**. Before downloading data, agree to the [NSD Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and submit the [NSD Data Access form](https://forms.gle/xue2bCdM9LaFNMeb7), as done in related NSD reconstruction repositories such as MindEye.

This repo does not automatically download NSD. Prepare the data locally and pass the paths to the scripts. The default expected layout is:

```text
data/
├── nsd/
│   ├── betas_all_subj01_fp32_renorm.hdf5
│   ├── betas_all_subj02_fp32_renorm.hdf5
│   └── wds/
│       └── subj01/
│           ├── train/
│           │   ├── 0.tar
│           │   └── ...
│           └── new_test/
│               ├── 0.tar
│               └── ...
├── images/
│   ├── image_000000.png
│   ├── image_000001.png
│   └── ...
└── annotations/
    └── COCO_73k_annots_curated.npy
```

Each WebDataset tar should contain the behavioral arrays used to index the beta HDF5 files and stimulus images:

```text
behav.npy
past_behav.npy
future_behav.npy
olds_behav.npy
```

The scripts fail early with a clear error if a required data path or checkpoint is missing.

## Checkpoints

Large files are intentionally not committed. Put released or locally trained checkpoints under:

```text
pretrained/
├── vqvae.safetensors
└── fmri_last.pth                # optional convenience copy of a Stage 1 checkpoint

outputs/
├── brainencoder/
└── mindhier_stage2/
```

`pretrained/`, `outputs/`, and `data/` are ignored by Git.

## Stage 1 Training

Train the hierarchical fMRI encoder for one subject:

Stage 1 uses command-line arguments in `brainencoder/train.py`; editing `configs/default.yaml` will not change this command.

```bash
python brainencoder/train.py \
  --data_path data/nsd \
  --images_dir data/images \
  --captions_path data/annotations/COCO_73k_annots_curated.npy \
  --output_dir outputs/brainencoder \
  --subj 1 \
  --no-multi_subject
```

The main output is:

```text
outputs/brainencoder/<run_name>/last-hierarchy-vitl.pth
```

This checkpoint is used as the fMRI conditioner for Stage 2.

## Stage 2 Training

Train the hierarchy-conditioned autoregressive generator:

Stage 2 uses command-line arguments in `MindHier/utils/arg_util.py`; editing `configs/default.yaml` will not change this command.

```bash
python MindHier/train.py \
  --data_path data/nsd \
  --images_dir data/images \
  --fmri_ckpt outputs/brainencoder/subj_vit_l_lr0.0001_layers12_16_20_24_subj1/last-hierarchy-vitl.pth \
  --vae_ckpt pretrained/vqvae.safetensors \
  --run_name subj1_stage2 \
  --subj 1
```

Stage 2 trains one subject at a time. The old multi-subject switch is kept only for legacy compatibility and is disabled for the clean paper reproduction path.

## Inference and Evaluation

The main inference code lives in:

```text
MindHier/models/pipeline.py
```

Evaluation utilities are provided in:

```text
brainencoder/eval.py
MindHier/calculate_metrics.py
```

Example metric command:

```bash
python brainencoder/eval.py \
  --results_path outputs/reconstructions/subj01
```

Reported metrics typically include pixel correlation, SSIM, CLIP-based similarity, and retrieval-style semantic metrics.

## Citation

```bibtex
@inproceedings{
  zhang2026moving,
  title={Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for f{MRI}-to-Image Reconstruction},
  author={Xu Zhang and Ruijie Quan and Wenguan Wang and Yi Yang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2510.22335},
  eprint={2510.22335},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

If you use NSD, please also cite the Natural Scenes Dataset paper.

## License

This code is released under the Apache-2.0 License. Please follow the NSD license and terms when using NSD-derived data.
