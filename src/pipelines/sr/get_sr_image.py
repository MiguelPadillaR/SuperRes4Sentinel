import os
import glob
import argparse
from pathlib import Path
import numpy as np
import time
import torch
import rasterio

from PIL import Image

from ...utils.constants import SR_5M_DIR
from ...utils.utils import make_grid
from .utils import percentile_stretch, stack_bgrn
from .L1BSR_wrapper import L1BSRSR  # wrapper that loads RCAN & runs SR

def save_rgb_png(sr, out_path):
    """Save SR result as stretched RGB PNG"""
    rgb = np.stack([sr[..., 2], sr[..., 1], sr[..., 0]], axis=-1)  # B04,R / B03,G / B02,B
    rgb_u8 = percentile_stretch(rgb)
    Image.fromarray(rgb_u8).save(out_path)

def save_multiband_tif(sr, reference_band, out_path):
    """Optionally save SR result as multiband GeoTIFF using metadata from reference band"""
    with rasterio.open(reference_band) as src:
        profile = src.profile
    profile.update(count=4, dtype=rasterio.uint16, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(sr[..., i], i + 1)

def make_comparison_grid(b02, b03, b04, sr_u16, out_path):
    # Original RGB (percentile stretched)
    orig_rgb_u8 = percentile_stretch(np.stack([b04, b03, b02], axis=-1))

    print("OG image shape:", orig_rgb_u8.shape)

    # Double it with NEAREST to make blocky pixels visible
    h, w, _ = orig_rgb_u8.shape
    factor_h = int(sr_u16.shape[0]) // h
    factor_w = int(sr_u16.shape[1]) // w
    orig_up = np.array(Image.fromarray(orig_rgb_u8).resize((w*factor_w, h*factor_h), Image.NEAREST))

    # SR RGB (percentile stretched, already at double size)
    sr_rgb_u8 = percentile_stretch(np.stack([sr_u16[...,2], sr_u16[...,1], sr_u16[...,0]], axis=-1))

    # Build comparison grid (original vs SR)
    grid = make_grid([orig_up, sr_rgb_u8], ncols=2)
    Image.fromarray(grid).save(out_path)

def save_comparison_grid(orig_rgb, sr_rgb, out_path):
    """Save side-by-side comparison grid between original and SR"""
    # Resize original to SR size
    h, w = sr_rgb.shape[:2]
    orig_resized = np.array(Image.fromarray(orig_rgb).resize((w, h), Image.BILINEAR))
    grid = make_grid([orig_resized, sr_rgb], ncols=2, pad=10)
    Image.fromarray(grid).save(out_path)

def process_directory(input_dir, output_dir):
    all_files = glob.glob(os.path.join(input_dir, "*.tif*"))
    groups = {}

    for f in all_files:
        base = os.path.basename(f)
        if "-" not in base:
            continue
        filename, ext = os.path.splitext(base)
        prefix, band = filename.rsplit("-", 1)
        if prefix not in groups:
            groups[prefix] = {}
        groups[prefix][band] = f

    for prefix, band_files in groups.items():
        missing = set(bands_order) - set(band_files.keys())
        if missing:
            print(f"Skipping {prefix}, missing bands: {missing}")
            continue

        b02 = rasterio.open(band_files["B02"]).read(1)
        b03 = rasterio.open(band_files["B03"]).read(1)
        b04 = rasterio.open(band_files["B04"]).read(1)
        b08 = rasterio.open(band_files["B08"]).read(1)

        # Get original RGB image
        h, w = b04.shape
        rgb_before_u8 = percentile_stretch(np.stack([b04, b03, b02], axis=-1))
        rgb_before_u8_resized = np.array(Image.fromarray(rgb_before_u8).resize((w*2, h*2), Image.NEAREST))

        # Save original (preview) PNG
        orig_png = os.path.join(SR_5M_DIR, f"lol.png")
        Image.fromarray(rgb_before_u8_resized).save(orig_png)

        # Stack input
        img_bgrn = stack_bgrn(
            type("Band", (), {"arr": b02}),
            type("Band", (), {"arr": b03}),
            type("Band", (), {"arr": b04}),
            type("Band", (), {"arr": b08}),
        )

        # Run SR
        sr_u16 = engine.super_resolve(img_bgrn)

        # Save PNG
        out_png = os.path.join(output_dir, f"{prefix}.png")
        save_rgb_png(sr_u16, out_png)
        print(f"Saved PNG: {out_png}")

        # Make and save comparison grid
        comp_png = COMP_DIR / f"{prefix}_comparison.png"
        grid = make_grid([rgb_before_u8_resized,
                        percentile_stretch(np.stack([sr_u16[...,2], sr_u16[...,1], sr_u16[...,0]], -1))],
                        ncols=2)
        Image.fromarray(grid).save(comp_png)
        print(f"Saved comparison grid: {comp_png}")

        # Optionally save TIF
        if args.tif:
            out_tif = os.path.join(output_dir, f"{prefix}.tif")
            save_multiband_tif(sr_u16, band_files["B02"], out_tif)
            print(f"Saved TIF: {out_tif}")
        print()

# --- Run ---
if __name__ == "__main__":
    start_time = time.time()
    CURR_SCRIPT_DIR = Path(__file__).resolve().parent
    # --- Ensure output dir exists ---
    OUT_DIR = SR_5M_DIR / "out"
    COMP_DIR = SR_5M_DIR / "comparison"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)


    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(
        description="Run L1BSR super-resolution on Sentinel-2 bands (B02,B03,B04,B08)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing Sentinel-2 band files with pattern <file_name>-Bxx.tiff",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUT_DIR,
        help="Output directory for results (default: RES_DIR/sr).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default= CURR_SCRIPT_DIR / "REC_Real_L1B.safetensors",
        help="Path to .safetensors model weights (default: REC_Real_L1B.safetensors).",
    )
    parser.add_argument(
        "--tif",
        action="store_true",
        help="Also save multiband SR result as GeoTIFF (alongside PNG).",
    )
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Required bands ---
    bands_order = ["B02", "B03", "B04", "B08"]

    # --- Load model ---
    
    engine = L1BSRSR(weights_path=args.weights, device=device)

    process_directory(args.input, args.output)

    finish_time = time.time()
