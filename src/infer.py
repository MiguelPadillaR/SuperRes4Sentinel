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


def run_inference(image_paths: List[Path], model_name='edsr', scale=4, weigths_filename: str=None, progressive_to: int=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig(name=model_name, scale=scale, pretrained=CKPT_DIR / weigths_filename)
    model = build_model(cfg).to(device)
    model.eval()

    RES_DIR.mkdir(parents=True, exist_ok=True)

    for im in image_paths:
        lr = load_image(im)

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
        compare_filename = f"{im.stem}-compare_LR-x{scale}.png"
        
        # Save scaled outputs
        imwrite(dirs[f"x{scale}"] / f"{im.stem}-SRx{scale}.png", sr_np)
        print("Saved upscaled image in ", RES_DIR / dirs[f"x{scale}"] / f"{im.stem}_SRx{scale}.png")
        imwrite(dirs["compare"] / compare_filename, grid)
        print("Saved comparison in ", RES_DIR / dirs["compare"] / compare_filename)
        
        # Save progressively scaled outputs
        if sr_p.any():
            imwrite(dirs[scale_factor] / f"{im.stem}-SR{scale_factor}.png", sr_p)
            print("Saved improved upscaled image in ", RES_DIR / dirs[scale_factor] / f"{im.stem}-SR{scale_factor}.png")

            lr_rsz = cv2.resize(lr, (sr_p.shape[1], sr_p.shape[0]), interpolation=cv2.INTER_CUBIC)
            sr_np_rsz = cv2.resize(sr_np, (sr_p.shape[1], sr_p.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            grid = make_grid([lr_rsz, sr_np_rsz, sr_p], ncols=3)
            compare_filename = f"{im.stem}-compare_LR-x{scale}-{scale_factor}.png"

            imwrite(dirs["compare"] / compare_filename, grid)
            print("Saved improved comparison in ", RES_DIR / dirs["compare"] / compare_filename)
            
        print()
    print("All done.")

if __name__ == '__main__':
    # Example usage:
    #   python -m src.infer data/LR/36.22243_-5.84406_test.png data/LR/36.31215_-5.93888_test.png --model edsr --scale 4 --weigths best_edsr_x4.pth --progressive-to 8
    #   python -m src.infer data/LR/*.png --model edsr --scale 4 --weigths best_edsr_x4.pth
    #   python -m src.infer data/LR --model edsr --scale 4 --weigths best_edsr_x4.pth
    
    parser = argparse.ArgumentParser(
        description="Run super-resolution inference on images or a directory of images.",
        epilog=(
            "USAGE:\n"
            "  # Run on two specific images\n"
            "  python -m src.infer data/LR/img1.png data/LR/img2.png --model edsr --scale 4 --weigths best_edsr_x4.pth\n\n"
            "  # Run on all images in a directory\n"
            "  python -m src.infer data/LR --model edsr --scale 4 --weigths best_edsr_x4.pth\n\n"
            "  # Progressive upscaling from x4 to x8\n"
            "  python -m src.infer data/LR/*.png --model edsr --scale 4 --progressive-to 8\n"
            " "
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'images',
        nargs='+',
        type=str,
        help=(
            "One or more image file paths (e.g. data/LR/img1.png img2.png). "
            "If a single directory is provided, all image files inside it will be processed."
        )
    )
    parser.add_argument(
        '--model',
        type=str,
        metavar='model_name',
        default='edsr',
        help="Super-resolution model to use (default: edsr)."
    )
    parser.add_argument(
        '--scale',
        type=int,
        metavar='sc_factor',
        default=4,
        help="Upscaling factor for the model (default: 4)."
    )
    parser.add_argument(
        '--weights',
        type=str,
        metavar='w_path',
        default="best_edsr_x4.pth",
        help="Path to the model checkpoint (.pth) file. If not provided, uses default weights."
    )
    parser.add_argument(
        '--progressive-to',
        type=int,
        metavar='sc_factor',
        default=4,
        help=(
            "Optionally upscale further in progressive steps (e.g. from x4 to x8). "
            "If set, the output will be generated up to this factor."
        )
    )
    
    args = parser.parse_args()

    # Convert to Path objects
    image_paths = [Path(p) for p in args.images]

    # If only one arg is passed and it's a directory → expand
    if len(image_paths) == 1 and image_paths[0].is_dir():
        image_paths = sorted([p for p in image_paths[0].iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])

    run_inference(image_paths, model_name=args.model, scale=args.scale, weigths_filename=args.weights, progressive_to=args.progressive_to)
