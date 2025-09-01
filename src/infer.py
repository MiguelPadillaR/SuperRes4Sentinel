import argparse

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from src.utils.constants import CKPT_DIR, RES_DIR
from src.utils.utils import imread, imwrite, make_grid
from src.model.model import ModelConfig, build_model


def load_image(path: Path):
    img = imread(path)
    return img


def run_inference(image_paths: List[Path], model_name='edsr', scale=4, ckpt_filename: str=None, progressive_to: int=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig(name=model_name, scale=scale, pretrained=CKPT_DIR / model_name / ckpt_filename)
    model = build_model(cfg).to(device)
    model.eval()

    RES_DIR.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        lr = load_image(p)

        with torch.no_grad(), torch.autocast("cuda"):
            x = to_tensor(lr).unsqueeze(0).to(device)
            sr = model(x).clamp(0,1)
            sr_np = (sr[0].permute(1,2,0).cpu().numpy()*255.0).round().astype(np.uint8)

        # Progressive beyond x4 if requested (e.g., to x8 or x10)
        if progressive_to and progressive_to > scale:
            sr_p = None
            factor = progressive_to / scale
            # Refeed the output to upscale further
            for _ in range(int(factor) - 1):
                x = to_tensor(sr_np).unsqueeze(0).to(device)
                sr = model(x).clamp(0,1)
                sr_p = (sr[0].permute(1,2,0).cpu().detach().numpy()*255.0).round().astype(np.uint8)
        
        # Create res dirs 
        scale_factor ='x' + (str(progressive_to) if progressive_to else str(scale))
        dirs = {
            "compare": Path(RES_DIR / "comparison"),
            f"x{scale}": Path(RES_DIR / f"x{scale}"),
            scale_factor : Path(RES_DIR / f"x{progressive_to}" if sr_p.any() else '')
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True) if len(str(d)) > 1 else None

        # Resize images for visual comparison
        lr_rsz = cv2.resize(lr, (sr_np.shape[1], sr_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        grid = make_grid([lr_rsz, sr_np], ncols=2)
        compare_filename = f"{p.stem}-compare_LR-x{scale}.png"
        
        # Save scaled outputs
        imwrite(dirs[f"x{scale}"] / f"{p.stem}-SRx{scale}.png", sr_np)
        print("Saved upscaled image in ", RES_DIR / dirs[f"x{scale}"] / f"{p.stem}_SRx{scale}.png")
        imwrite(dirs["compare"] / compare_filename, grid)
        print("Saved comparison in ", RES_DIR / dirs["compare"] / compare_filename)
        
        # Save progressively scaled outputs
        if sr_p.any():
            lr_rsz = cv2.resize(lr, (sr_p.shape[1], sr_p.shape[0]), interpolation=cv2.INTER_CUBIC)
            sr_np_rsz = cv2.resize(sr_np, (sr_p.shape[1], sr_p.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            grid = make_grid([lr_rsz, sr_np_rsz, sr_p], ncols=3)
            compare_filename = f"{p.stem}-compare_LR-x{scale}-{scale_factor}.png"

            imwrite(dirs["compare"] / compare_filename, grid)
            print("Saved improved comparison in ", RES_DIR / dirs["compare"] / compare_filename)
            
            imwrite(dirs[scale_factor] / f"{p.stem}-SR{scale_factor}.png", sr_p)
            print("Saved improved upscaled image in ", RES_DIR / dirs[scale_factor] / f"{p.stem}-SR{scale_factor}.png")


if __name__ == '__main__':
    # Example usage:
    # python -m src.infer data/LR/36.22243_-5.84406_test.png data/LR/36.31215_-5.93888_test.png --model edsr --scale 4 --ckpt best_edsr_x4.pth --progressive-to 8
    
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+', type=str)
    parser.add_argument('--model', type=str, default='edsr')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--progressive-to', type=int, default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    run_inference(paths, model_name=args.model, scale=args.scale, ckpt_filename=args.ckpt, progressive_to=args.progressive_to)
