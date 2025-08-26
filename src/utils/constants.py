from pathlib import Path

# Paths (edit as needed)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
LR_DIR = DATA_DIR / 'LR'
HR_DIR = DATA_DIR / 'HR'
OUT_DIR = ROOT / 'out'
CKPT_DIR = OUT_DIR / 'checkpoints'
RES_DIR = OUT_DIR / 'res'

# Training / preprocessing
RANDOM_SEED = 42
TILE_SIZE_HR = 128  # crop size on HR; LR crop will be TILE_SIZE_HR // SCALE
SCALE = 4           # 2 or 4 recommended for training
BATCH_SIZE = 8
NUM_WORKERS = 4
LR_INIT = 2e-4
EPOCHS = 40
VAL_INTERVAL = 1

# Model
MODEL_NAME = 'edsr'  # 'edsr' | 'esrgan' | 'srcnn'
NUM_RESBLOCKS = 16   # for EDSR
N_FEATS = 64         # for EDSR

# Inference
SAVE_COMPARE_GRID = True
