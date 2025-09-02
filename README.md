# Super-resolution for Sentinel RGB (SuperRes4Sentinel)
The SuperRes4Sentinel module has been developed as part of the image enhancing pipeline for KHAOS research group's Agricultural Imaging Assistant (AgrIA) project.

## Features:
- Enhancing of true color RGB pictures from Sentinel.

## Installation
### Requirements
- **A CONDA environment is heavily encouraged**, as it usually is more solid and problems are easier to pinpoint, but **Python's standard virtual environments will also do** with minor adjustments.
- All `.env` related data, including:
    - **A Copernicus DataSpace Ecosystem account and its credentials for remote usage**. It's free to create and tutorials for token generation are available.
    - **A Google Maps Static API key**. Really, any Google project with an API key will do, the usage of GMS is well within the free tier for minor scales experimentation.

### Setup


## Quickstart
Most of the modules run as scripts. Each one has a brief general description and parameters overview using the `-h` or `--help` option. All of them run on default parameters when no options values are specified. Most of these values can be found and modified in the `constants.py` file, but we recommeno you tweak it in the options directly.

All of these scripts are run from root:
```
cd SuperRes4Sentinel/
```
### Image retrieval
In order to build a model, you will need images to train it with. Use the following to get 500 image pairs to train with:
```bash
python -m src.pipelines.get_image_pairs
```
### Training
Use the following to build an EDSR model with a x4 scaling factor. Best model will be selected from among 500 epochs:
```bash
python -m src.train
```
### Inference
Whenever you are ready to try your model, you will need to indicate the positional argument for `images`, wither with one or several filepaths of the images to super-resolve or a dir contaitning such images:
```bash
python -m src.infer path/to/your/LR/images
```
## Project structure
After a complete execution of all project's features (image retrieval, training and inference) this will be the project's resulting file tree:
```
(2025-08-28)

SuperRes4Sentinel
├── data                        # Stores the training images. They must be true color images (png, jpg, tiff...).
│   ├── bands                   # Saves Sentinel's band images.
│   │   ├── ...png,jpg,tiff...
│   │   └── ...
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
│   ├── checkpoints             # Contains files with the best model weights as well as in different stages of training by epochs.
│   │   ├── ...pth
│   │   └── ...
│   └── res                     # Contains the SR images results after inference.
│       ├── comparison          # Svaes all comparison of both LR-x{scale} and LR-x{scale}-x{prog_scale} of each image's upscaling.
│       │   ├── ...png
│       │   └── ...
│       ├── x{scale1}           # Saves high resolution images' upscaling results at {scale1}.
│       │   ├── ...png
│       │   └── ...
│       ├── x{scale2}           # Saves high resolution images' upscaling results at {scale2}.
│       │   ├── ...png
│       │   └── ...
│       └── ...
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