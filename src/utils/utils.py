import math
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image


def imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(img2, img1, data_range=255))


def get_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # multichannel=True is deprecated; channel_axis=2
    return float(structural_similarity(img2, img1, data_range=255, channel_axis=2))


def make_grid(images, ncols=3, pad=4) -> np.ndarray:
    """Make a grid of images (HxWx3 uint8) with padding."""
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    norm = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for im in images]
    n = len(norm)
    rows = math.ceil(n / ncols)
    grid = np.ones(((h+pad)*rows+pad, (w+pad)*ncols+pad, 3), dtype=np.uint8) * 255
    for idx, im in enumerate(norm):
        r, c = divmod(idx, ncols)
        y, x = pad + r*(h+pad), pad + c*(w+pad)
        grid[y:y+h, x:x+w] = im
    return grid
