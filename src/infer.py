from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from constants import OUT_DIR, RES_DIR
from model import ModelConfig, build_model
from utils import imread, imwrite, make_grid, to_uint8, resize


def load_image(path: Path):
    img = imread(path)
    return img


def run_inference(image_paths: List[Path], model_name='edsr', scale=4, ckpt_path: str=None, progressive_to: int=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig(name=model_name, scale=scale, pretrained=ckpt_path)
    model = build_model(cfg).to(device)
    model.eval()

    RES_DIR.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        lr = load_image(p)
        # bicubic upscale to target scale for baseline
        bic = cv2.resize(lr, (lr.shape[1]*scale, lr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

        with torch.no_grad():
            x = to_tensor(lr).unsqueeze(0).to(device)
            sr = model(x).clamp(0,1)
            sr_np = (sr[0].permute(1,2,0).cpu().numpy()*255.0).round().astype(np.uint8)

        # Progressive beyond x4 if requested (e.g., to x8 or x10)
        if progressive_to and progressive_to > scale:
            factor = progressive_to / scale
            sr_np = resize(sr_np, factor)

        grid = make_grid([lr, bic, sr_np], ncols=3)
        imwrite(RES_DIR / f'{p.stem}_compare.png', grid)
        imwrite(RES_DIR / f'{p.stem}_SR.png', sr_np)
        print('Saved', RES_DIR / f'{p.stem}_SR.png')


if __name__ == '__main__':
    # Example usage:
    # python -m src.infer data/LR/sample1.png data/LR/sample2.png --model esrgan --scale 4 --ckpt weights.pth --progressive_to 8
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+', type=str)
    parser.add_argument('--model', type=str, default='edsr')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--progressive_to', type=int, default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    run_inference(paths, model_name=args.model, scale=args.scale, ckpt_path=args.ckpt, progressive_to=args.progressive_to)
