# Super-resolution for Sentinel RGB (experimental)

- Train EDSR ×4 on paired Sentinel-2 (LR) and Google Maps (HR) crops.
- Evaluate PSNR/SSIM and visualize LR vs Bicubic vs SR.
- Optionally use ESRGAN/RRDBNet weights for sharper textures.

### Quick start
1. Put paired images in `data/LR` and `data/HR` with **matching filenames**.
2. `pip install -r requirements.txt`
3. Edit `src/constants.py` (SCALE, TILE_SIZE_HR, etc.)
4. `python -m src.train`
5. `python -m src.infer data/LR/your_image.png --model edsr --scale 4 --ckpt outputs/checkpoints/best_edsr_x4.pth --progressive_to 8`

> To approach 10 m→1 m (×10), train at ×4 and upscale progressively (×4→×8→resize to ×10) while you experiment. True ×10 fidelity usually benefits from multi-stage or diffusion-based models and well-registered training data.

---

## Notes on weights & articles to try next

- **EDSR** (baseline, stable for ×2/×4). Train from scratch on your tiles; convergence is predictable with L1 loss.
- **ESRGAN / RRDBNet** (perceptual, sharper). Start from public ×4 RRDBNet weights, then finetune on your domain to reduce artifacts over fields/rooftops.
- **Progressive SR** to push beyond ×4: apply model multiple times (×2→×4→×8) and finish with high-quality resize to reach ×10.