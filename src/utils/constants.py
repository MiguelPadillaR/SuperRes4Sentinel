from pathlib import Path

# Paths (edit as needed)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
BAND_DIR = DATA_DIR / 'bands'
LR_DIR = DATA_DIR / 'LR'
HR_DIR = DATA_DIR / 'HR'
OUT_DIR = ROOT / 'out'
CKPT_DIR = OUT_DIR / 'checkpoints'
RES_DIR = OUT_DIR / 'res'
SR_5M_DIR = RES_DIR / "sr_5m"

# Paired images retrieval constants
DELTA_DAYS = 10
LAT_MIN, LAT_MAX = 37.230328, 43.109004  # Andalusia, Spain: 36.125397, 38.682026
LON_MIN, LON_MAX = -8.613281, -1.878662  # Andalusia, Spain: -7.344360, -1.796265
SIZE = 500 # 255  # image size in pixels (width, height)
TILE_SIZE_ESRI = 256  # ESRI tiles are always 256x256 px
ZOOM = 17   # zoom level for Google Maps, ESRI and Sentinel-2

REFLECTANCE_SCALE = 400.0

# Training / preprocessing
RANDOM_SEED = 42
TILE_SIZE_HR = 96  # crop size on HR; LR crop will be TILE_SIZE_HR // SCALE
SCALE = 4           # 2 or 4 recommended for training
BATCH_SIZE = 8
N_WORKERS = 4
LR_INIT = 2e-4
EPOCHS = 500
VAL_INTERVAL = 1

# Model
MODEL_NAME = 'edsr'  # 'edsr' | 'esrgan' | 'srcnn'
N_RESBLOCKS = 4   # for EDSR
N_FEATS = 32         # for EDSR

# Inference
SAVE_COMPARE_GRID = True
