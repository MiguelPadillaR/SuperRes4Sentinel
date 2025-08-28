# Super-resolution for Sentinel RGB (SuperRes4Sentinel)
The SuperRes4Sentinel module has been developed as part of the image enhancing pipeline for KHAOS research group's Agricultural Imaging Assistant (AgrIA) project.

## Features:
- Enhancing of true color RGB pictures from Sentinel.

## Requirements
- 

## Installation

## Quickstart

## Project structure
The following display explains the project's file tree:
```
(2025-08-28)

SuperRes4Sentinel
├── data                        # Stores the training images. They must be true color images (png, jpg, tiff...).
│   ├── HR                      # Saves high resolution images from Google Maps' ground truth.
│   │   ├── ...png
│   │   └── ...
│   └── LR                      # Saves low resolution images from Sentinel.
│       ├── ...png
│       └── ...
│
│
├── notebooks                   #  (IN PROGRESS) Stores different notebooks to test different parts of the project.
│   ├── 00_introduction.ipynb
│   └── 01_quickstart.ipynb
│
│
├── out                         # Stores model output.
│   ├── checkpoints             # Contains files with the best models as well as the model in different stages of training by epochs.
│   │   ├── ...pth
│   │   └── ...
│   └── res                     # Contains the SR images results after inference.
│       ├── ...png
│       └── ...
│
│
├── src                         # Stores all organised source code and the bulk of the programming.
│   ├── data                    # Contains scripts that set up the model's training dataset
│   │   └── dataset.py          # Holds all methods for instances of `PairedDataset` objects to be generated and setup for later usage.
│   │
│   ├── model                   # Contains all code related to SR model building.
│   │   └── model.py            # Keeps default config setup and factory method for SR models.
│   │
│   ├── pipelines               # (NAME UNCLEAR) Contains model training pipeline related scripts and setup.
│   │   ├── get_image_pairs.py  # Keeps methods to download and pair images from Sentinel and Google.
│   │   ├── sh_config.py        # Initializes the Sentinel Hub configuration for image retrieval using Copernicus credentials.
│   │   └── utils.py            # Helper methods for downloading process.
│   │
│   ├── utils                   # Stores an assorment of helper functions and project constants.
│   │   ├── constants.py        # Contains all project's most relevant constants that are used all trhoughout.
│   │   └── utils.py            # Helper methods for model's training and image inference.
│   ├── infer.py                # Image's inference workflow.
│   └── train.py                # Model's training workflow.
│
├── LICENSE
├── README.md
└── requirements.txt

```

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