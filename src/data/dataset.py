from pathlib import Path
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.constants import TILE_SIZE_HR, SCALE, RANDOM_SEED
from src.utils.utils import imread

random.seed(RANDOM_SEED)


def list_image_paths(root: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


class PairedImageDataset(Dataset):
    """
    Performs random HR crops (size TILE_SIZE_HR), with aligned LR crops.\n
    Assumes *matching filenames* between LR and HR folders.
    Example: data/LR/sceneA.png  <-> data/HR/sceneA.png
    """
    def __init__(self, lr_dir: Path, hr_dir: Path, scale: int = SCALE, augment: bool = True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.augment = augment

        hr_paths = list_image_paths(self.hr_dir)
        self.pairs: List[Tuple[Path, Path]] = []
        for hp in hr_paths:
            rel = hp.relative_to(self.hr_dir)
            lp = (self.lr_dir / rel).with_suffix(hp.suffix)
            if lp.exists():
                self.pairs.append((lp, hp))
        if not self.pairs:
            raise RuntimeError('No LR-HR pairs found. Ensure matching filenames in LR/ and HR/.')

    def __len__(self):
        return len(self.pairs)

    def _rand_crop_coords(self, h: int, w: int, size: int) -> Tuple[int,int]:
        if h <= size or w <= size:
            return 0, 0
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return y, x

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # flips and 90-degree rotations
        if random.random() < 0.5:
            lr = np.flip(lr, axis=1); hr = np.flip(hr, axis=1)
        if random.random() < 0.5:
            lr = np.flip(lr, axis=0); hr = np.flip(hr, axis=0)
        k = random.randint(0, 3)
        if k:
            lr = np.rot90(lr, k); hr = np.rot90(hr, k)
        return lr.copy(), hr.copy()
    
    def _get_rand_crop(self, im: np.ndarray, size_hr: int) -> np.ndarray:
        H, W = im.shape[:2]
        y, x = self._rand_crop_coords(H, W, size_hr)
        return im[y:y+size_hr, x:x+size_hr]

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        lr = imread(lp)
        hr = imread(hp)
        # ensure shapes are multiples of scale
        size_hr = TILE_SIZE_HR
        
        hr_crop = self._get_rand_crop(hr, size_hr)
        lr_crop = self._get_rand_crop(hr, size_hr//self.scale)

        # lr_crop = cv2.resize(hr_crop, (size_hr//self.scale, size_hr//self.scale), interpolation=cv2.INTER_AREA)
        # Replace lr_crop with real LR if available and aligned
        # Here we assume perfect registration and simply crop+downsample from HR for stability
        # To use provided LR tiles directly, comment the above and instead crop from lr with scaled coords.

        if self.augment:
            lr_crop, hr_crop = self._augment(lr_crop, hr_crop)

        # to tensor (C,H,W), range [0,1]
        lr_t = torch.from_numpy(lr_crop.transpose(2,0,1)).float() / 255.0
        hr_t = torch.from_numpy(hr_crop.transpose(2,0,1)).float() / 255.0
        return {"lr": lr_t, "hr": hr_t}

