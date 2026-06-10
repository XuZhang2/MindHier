import webdataset as wds
from webdataset import WebLoader
import h5py
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils.constants import DEFAULT_MULTI_SUBJECTS, DEFAULT_TEST_SAMPLES, NSD_TRIALS_PER_SESSION

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def normalize_255_into_pm1(x):
    return x.div_(127.5).add_(-1)

class NSDBrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        images_dir: str,  # New parameter for the directory containing PNG images
        final_reso: int,
        subj: int = 1,
        num_sessions: int = 40,
        multi_subject: bool = False,
        max_samples: int = None,
        is_train: bool = True,
        mid_reso: float = 1.125,
    ):
        """
        Args:
            data_path: Path to dataset root
            images_dir: Path to directory containing PNG images (image_000000.png to image_072999.png)
            final_reso: Final image resolution
            subj: Subject ID (1-8)
            num_sessions: Number of sessions to use
            multi_subject: Whether to use multiple subjects
            max_samples: Maximum number of samples to use (for debugging)
            is_train: Whether this is training data (affects session selection)
        """
        self.data_path = data_path
        self.images_dir = images_dir
        self.max_samples = max_samples
        self.is_train = is_train
        self.subj = subj
        self.multi_subject = multi_subject
        self.final_reso = final_reso
        
        # Determine subject list
        self.subj_list = DEFAULT_MULTI_SUBJECTS if multi_subject else [subj]
        
        mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
        
        # Image preprocessing
        self.train_transform = transforms.Compose([
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            # transforms.CenterCrop((final_reso, final_reso)),
            # transforms.Resize((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1
        ])
        self.test_transform = transforms.Compose([
            # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
            # transforms.RandomCrop((final_reso, final_reso)),
            transforms.Resize((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1
        ])
        
        # Load voxel data for all subjects
        self.voxels = {}
        self.num_voxels_list = []
        for s in self.subj_list:
            betas_path = os.path.join(data_path, f'betas_all_subj0{s}_fp32_renorm.hdf5')
            if not os.path.isfile(betas_path):
                raise FileNotFoundError(
                    f"Missing beta file: {betas_path}. "
                    "Prepare NSD betas or set --data_path to the preprocessed NSD root."
                )
            with h5py.File(betas_path, 'r') as f:
                self.voxels[f'subj0{s}'] = torch.tensor(f['betas'][:], dtype=torch.float32).cpu()
            self.num_voxels_list.append(self.voxels[f'subj0{s}'].shape[-1])
        
        # Create WebDataset for each subject
        self.datasets = {}
        self.sample_counts = {}
        total_samples = 0
        
        for s in self.subj_list:
            session_range = "0..39" if (multi_subject or num_sessions == 40) else f"0..{num_sessions-1}"
            split = "train" if is_train else "new_test"
            wds_dir = os.path.join(data_path, "wds", f"subj0{s}", split)
            if not os.path.isdir(wds_dir):
                raise FileNotFoundError(
                    f"Missing WebDataset split directory: {wds_dir}. "
                    "Expected tar shards such as 0.tar, 1.tar, ..."
                )
            url = f"{wds_dir}/{{{session_range}}}.tar"
            
            # Create dataset
            ds = wds.WebDataset(url, resampled=is_train, nodesplitter=lambda urls: urls)
            if is_train:
                ds = ds.shuffle(NSD_TRIALS_PER_SESSION, initial=2 * NSD_TRIALS_PER_SESSION, rng=random.Random(42))
            ds = (
                ds.decode("torch")
                .rename(
                    behav="behav.npy",
                    past_behav="past_behav.npy",
                    future_behav="future_behav.npy",
                    olds_behav="olds_behav.npy"
                )
                .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
            )
            
            # Get sample count (approximate for training with resampling)
            if is_train:
                # Training uses resampling, so we'll use a fixed epoch size
                sample_count = NSD_TRIALS_PER_SESSION * num_sessions
            else:
                # Default NSD held-out split size used by the paper.
                sample_count = DEFAULT_TEST_SAMPLES
            
            if max_samples is not None:
                sample_count = min(sample_count, max_samples)
            
            self.sample_counts[f'subj0{s}'] = sample_count
            total_samples += sample_count
            self.datasets[f'subj0{s}'] = ds
        self.dataset_iters = {}
        for key in self.datasets:
            self.dataset_iters[key] = iter(WebLoader(self.datasets[key], batch_size=1, shuffle=False))
            
        self.total_samples = total_samples
        
        # Pre-build indices for deterministic access
        self.indices = []
        for s in self.subj_list:
            key = f'subj0{s}'
            self.indices.extend([(key, i) for i in range(self.sample_counts[key])])
        
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
            self.total_samples = len(self.indices)

    def __len__(self):
        return self.total_samples

    def load_image(self, image_idx):
        """Load an image from the images directory using the index"""
        # Format the image filename with leading zeros
        image_filename = f"image_{image_idx:06d}.png"
        image_path = os.path.join(self.images_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}. "
                "Set --images_dir to your extracted NSD image directory."
            )
        
        # Load and transform the image
        with Image.open(image_path) as img:
            if self.is_train:
                image = self.train_transform(img)
            else:
                image = self.test_transform(img)
        
        return image.float()
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        subj_key, subj_idx = self.indices[idx]

        try:
            sample = next(self.dataset_iters[subj_key])
        except StopIteration:
            self.dataset_iters[subj_key] = iter(WebLoader(self.datasets[subj_key], batch_size=1, shuffle=False))
            sample = next(self.dataset_iters[subj_key])
        sample = sample[0]

        behav = sample[0]  # (behav, past_behav, future_behav, olds_behav)

        image_idx = int(behav[0, 0])
        voxel_idx = int(behav[0, 5])

        image = self.load_image(image_idx)
        voxel = self.voxels[subj_key][voxel_idx]

        return image, voxel


    # def __getitem__(self, idx):
    #     if idx >= len(self):
    #         raise IndexError
            
    #     # Get which subject and which index within that subject
    #     subj_key, subj_idx = self.indices[idx]
        
    #     # Get the sample from WebDataset
    #     sample = next(iter(self.datasets[subj_key].with_epoch(1).with_length(1)))
    #     behav = sample[0]  # (behav, past_behav, future_behav, olds_behav)
        
    #     # Get image and voxel data
    #     image_idx = int(behav[0, 0])
    #     voxel_idx = int(behav[0, 5])
        
    #     # Load image from PNG file instead of HDF5
    #     image = self.load_image(image_idx)
    #     voxel = self.voxels[subj_key][voxel_idx]
               
    #     # Return as (image, voxel) pair
    #     return image, voxel

def coco_collate_fn(batch):
    """Collate function that handles both images and voxels"""
    images = torch.stack([item[0] for item in batch])
    voxels = torch.stack([item[1] for item in batch])
    return images, voxels

def build_dataset(
    data_path: str,
    images_dir: str,  # Added new parameter
    final_reso: int,
    subj: int = 1,
    num_sessions: int = 40,
    multi_subject: bool = False,
    max_samples: int = None,
    is_train: bool = True,
    mid_reso=1.125,
):
    """Builds a dataset compatible with your DataLoader setup"""
    return NSDBrainDataset(
        data_path=data_path,
        images_dir=images_dir,  # Pass the new parameter
        final_reso=final_reso,
        subj=subj,
        num_sessions=num_sessions,
        multi_subject=multi_subject,
        max_samples=max_samples,
        is_train=is_train,
        mid_reso=mid_reso
    )
