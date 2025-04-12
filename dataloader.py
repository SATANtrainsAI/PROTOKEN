import os
import random
import math
import hashlib
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as T

# -------------------------
# 1) Utility: infinite loader
# -------------------------
def cycle(dl):
    """
    Creates an infinite generator from a DataLoader.
    """
    while True:
        for data in dl:
            yield data

# -------------------------
# 2) Custom collate function for zero-padding
# -------------------------
def next_multiple_of(value, multiple):
    """
    Round 'value' up to the next integer multiple of 'multiple'.
    Example: next_multiple_of(61, 16) = 64
    """
    return ((value + multiple - 1) // multiple) * multiple

def collate_with_padding(batch, patch_size):
    """
    Given a list of images (each a tensor of shape (3, H, W)), finds the largest
    spatial dimension in the batch, rounds it up to a multiple of patch_size, and
    zero-pads all images accordingly. Returns a single stacked tensor.
    """
    # Remove any items that failed to load.
    batch = [img for img in batch if img is not None]
    if len(batch) == 0:
        return None
    max_h = max(img.shape[1] for img in batch)
    max_w = max(img.shape[2] for img in batch)
    L = max(max_h, max_w)
    L_pad = next_multiple_of(L, patch_size)
    padded = []
    for img in batch:
        c, h, w = img.shape
        pad_h = L_pad - h
        pad_w = L_pad - w
        # F.pad takes (left, right, top, bottom)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        padded.append(padded_img)
    return torch.stack(padded, dim=0)

# -------------------------
# 3) Deterministic subset assignment
# -------------------------
def assign_subset(path, valid_frac):
    """
    Returns 'val' if a deterministic hash of the file path is less than valid_frac,
    otherwise 'train'. This ensures each file is always assigned to the same subset.
    """
    h = hashlib.md5(str(path).encode('utf-8')).hexdigest()
    h_int = int(h, 16)
    ratio = h_int / (2**128)
    return 'val' if ratio < valid_frac else 'train'

# -------------------------
# 4) Iterable Datasets with Global (DDP) and Worker Sharding
# -------------------------
class IterableImageDataset(IterableDataset):
    """
    Streams images from a single bin folder (e.g. /scratch/mnaseri1/img/bin_40_64)
    without preloading all file paths into memory.
    
    Optionally, yields only images assigned to a given subset ('train' or 'val')
    based on a deterministic hash and valid_frac.
    """
    def __init__(self, folder,
                 exts=['jpg', 'jpeg', 'png'],
                 normalize_mean=(0.4081, 0.5336, 0.4414),
                 normalize_std=(0.2538, 0.2752, 0.2157),
                 shuffle_buffer_size=10000,
                 subset=None,       # 'train', 'val', or None
                 valid_frac=0.0):
        super().__init__()
        self.folder = Path(folder)
        self.exts = set(ext.lower() for ext in exts)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        self.shuffle_buffer_size = shuffle_buffer_size
        self.subset = subset
        self.valid_frac = valid_frac
        print(f"Streaming images from {self.folder} for subset: {self.subset}")

    def _iter_paths(self):
        # Use os.walk to yield file paths one by one.
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.split('.')[-1].lower() in self.exts:
                    yield Path(root) / file

    def _stream_shuffle(self, iterator):
        # A simple streaming shuffle with a fixed-size buffer.
        buffer = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                for elem in buffer:
                    yield elem
                buffer = []
        if buffer:
            random.shuffle(buffer)
            yield from buffer

    def __iter__(self):
        # Global distributed sharding:
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # Worker info for DataLoader workers.
        worker_info = torch.utils.data.get_worker_info()

        paths_iter = self._iter_paths()
        paths_iter = self._stream_shuffle(paths_iter)
        if self.subset in ['train', 'val']:
            paths_iter = (p for p in paths_iter if assign_subset(p, self.valid_frac) == self.subset)
        # Global sharding: each distributed process gets every world_size-th file.
        paths_iter = (p for i, p in enumerate(paths_iter) if i % world_size == rank)

        # Worker-level sharding: if using multiple workers, split the iterator.
        if worker_info is not None:
            def worker_shard():
                for i, path in enumerate(paths_iter):
                    if i % worker_info.num_workers == worker_info.id:
                        yield self._load_image(path)
            return worker_shard()
        else:
            return (self._load_image(p) for p in paths_iter)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __getitem__(self, index):
        raise NotImplementedError("IterableImageDataset does not support __getitem__.")

class IterableImageDatasetUni(IterableDataset):
    """
    Streams images from all bins under a parent folder (excluding a given set)
    without loading all file paths into memory.
    
    Optionally splits images into 'train' and 'val' subsets based on a deterministic hash.
    """
    def __init__(self, folder,
                 exts=['jpg', 'jpeg', 'png'],
                 normalize_mean=(0.4081, 0.5336, 0.4414),
                 normalize_std=(0.2538, 0.2752, 0.2157),
                 shuffle_buffer_size=1000,
                 subset=None,
                 valid_frac=0.0):
        super().__init__()
        self.folder = Path(folder)
        self.exts = set(ext.lower() for ext in exts)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        self.shuffle_buffer_size = shuffle_buffer_size
        self.subset = subset
        self.valid_frac = valid_frac
        # Exclude these bins (if needed):
        self.yes = {
        Path("/scratch/mnaseri1/img/bin_321_384"), Path("/scratch/mnaseri1/img/bin_385_448"),
        Path("/scratch/xwang213/img/bin_321_384"), Path("/scratch/xwang213/img/bin_385_448")
        }

          #  yes list:  Path("/scratch/mnaseri1/img/bin_193_256"), Path("/scratch/mnaseri1/img/bin_257_320"), Path("/scratch/mnaseri1/img/bin_321_384"), Path("/scratch/mnaseri1/img/bin_385_448"), Path("/scratch/mnaseri1/img/bin_449_512")
          #  yes list:  Path("/scratch/xwang213/img/bin_193_256"), Path("/scratch/xwang213/img/bin_257_320"), Path("/scratch/xwang213/img/bin_321_384"), Path("/scratch/xwang213/img/bin_385_448"), Path("/scratch/xwang213/img/bin_449_512")

        print(f"Streaming images from {self.folder} (excluding specified bins) for subset: {self.subset}")

    def _iter_paths(self):
        bin_folders = sorted(self.folder.iterdir(), key=lambda p: str(p).lower())
        for bin_folder in bin_folders:
            if bin_folder.is_dir() and bin_folder in self.yes:
                print(f"Now streaming from bin: {bin_folder}")
                for root, _, files in os.walk(bin_folder):
                    for file in files:
                        if file.split('.')[-1].lower() in self.exts:
                            yield Path(root) / file


    def _stream_shuffle(self, iterator):
        buffer = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                for elem in buffer:
                    yield elem
                buffer = []
        if buffer:
            random.shuffle(buffer)
            yield from buffer

    def __iter__(self):
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker_info = torch.utils.data.get_worker_info()

        paths_iter = self._iter_paths()
        paths_iter = self._stream_shuffle(paths_iter)
        if self.subset in ['train', 'val']:
            paths_iter = (p for p in paths_iter if assign_subset(p, self.valid_frac) == self.subset)
        paths_iter = (p for i, p in enumerate(paths_iter) if i % world_size == rank)

        if worker_info is not None:
            def worker_shard():
                for i, path in enumerate(paths_iter):
                    if i % worker_info.num_workers == worker_info.id:
                        yield self._load_image(path)
            return worker_shard()
        else:
            return (self._load_image(p) for p in paths_iter)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __getitem__(self, index):
        raise NotImplementedError("IterableImageDatasetUni does not support __getitem__.")

# -------------------------
# 5) Training DataLoader Builders
# -------------------------
def train_for_bin(bin_path, patch_size, batch_size, ddp, ddp_world_size, ddp_rank, num_workers, valid_frac=0.0):
    """
    For a single bin folder, creates IterableImageDataset instances for training and validation (if valid_frac > 0).
    Note: DistributedSampler is not used with IterableDataset; sharding is handled inside the dataset.
    """
    if valid_frac > 0:
        train_ds = IterableImageDataset(folder=bin_path, subset='train', valid_frac=valid_frac)
        val_ds = IterableImageDataset(folder=bin_path, subset='val', valid_frac=valid_frac)
    else:
        train_ds = IterableImageDataset(folder=bin_path)
        val_ds = None

    train_dl = cycle(DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
    ))
    if val_ds is not None:
        val_dl = cycle(DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
        ))
    else:
        val_dl = None

    return train_dl, val_dl

def train_for_all(folder, patch_size, batch_size, ddp, ddp_world_size, ddp_rank, num_workers, valid_frac=0.0):
    """
    For a parent folder, creates IterableImageDatasetUni instances for streaming images from all bins (excluding a given set)
    with optional train/val splitting.
    """
    if valid_frac > 0:
        train_ds = IterableImageDatasetUni(folder=folder, subset='train', valid_frac=valid_frac)
        val_ds = IterableImageDatasetUni(folder=folder, subset='val', valid_frac=valid_frac)
    else:
        train_ds = IterableImageDatasetUni(folder=folder)
        val_ds = None

    train_dl = cycle(DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
    ))
    if val_ds is not None:
        val_dl = cycle(DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
        ))
    else:
        val_dl = None

    return train_dl, val_dl
