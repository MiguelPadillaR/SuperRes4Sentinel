import math
from pathlib import Path
from typing import Tuple

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


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def get_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(img2, img1, data_range=255))


def get_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # multichannel=True is deprecated; channel_axis=2
    return float(structural_similarity(img2, img1, data_range=255, channel_axis=2))


def make_grid(images, ncols=3, pad=4) -> np.ndarray:
    # images: list of HxWx3 uint8
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    norm = [cv2.copyMakeBorder(im, 0, h-im.shape[0], 0, w-im.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for im in images]
    n = len(norm)
    rows = math.ceil(n / ncols)
    grid = np.ones(((h+pad)*rows+pad, (w+pad)*ncols+pad, 3), dtype=np.uint8)*255
    for idx, im in enumerate(norm):
        r = idx // ncols
        c = idx % ncols
        y = pad + r*(h+pad)
        x = pad + c*(w+pad)
        grid[y:y+h, x:x+w] = im
    return grid


def center_crop_mod(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    y0 = (h - h2)//2
    x0 = (w - w2)//2
    return img[y0:y0+h2, x0:x0+w2]


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def register_images(img_lr, img_hr):
    # Convert to grayscale
    gray_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2GRAY)
    gray_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_lr, None)
    kp2, des2 = orb.detectAndCompute(gray_hr, None)

    # Match features.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography.
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    # Warp HR image to align with LR.
    h, w = img_lr.shape[:2]
    registered = cv2.warpPerspective(img_hr, H, (w, h))

    return registered
