"""Central configuration for the Intel Image Classification project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
VAL_FRACTION = 0.15  # share of seg_train held out for validation
