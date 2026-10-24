"""Central path and constant definitions, so scripts/notebooks don't hardcode paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRASHNET_DIR = DATA_RAW_DIR / "dataset-resized"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

RANDOM_SEED = 42

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLITS_MANIFEST_PATH = DATA_PROCESSED_DIR / "splits.csv"

# Input resolution used across every model (CNN-from-scratch and all transfer
# learning backbones) so results are comparable and pretrained ImageNet
# backbones get the input size they expect.
IMG_SIZE = 224

# ImageNet channel statistics — used to normalize inputs for the pretrained
# transfer-learning backbones (MobileNetV2/EfficientNet-B0/ResNet50 all
# expect this normalization). We reuse it for the from-scratch CNN too so
# every model sees identically preprocessed inputs and results stay comparable.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default training hyperparameters (from-scratch CNN, Phase 3)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

CNN_CHECKPOINT_PATH = MODELS_DIR / "cnn_from_scratch_best.pt"

# Transfer learning (Phase 4)
TRANSFER_ARCHITECTURES = ["mobilenet_v2", "efficientnet_b0", "resnet50"]

# Stage A (feature extraction, frozen backbone) can use a relatively large LR
# since only the freshly-initialized classifier head is training.
STAGE_A_LR = 1e-3
# Stage B (fine-tuning unfrozen upper layers) uses a much smaller LR — see
# src/models/transfer.py docstring for why.
STAGE_B_LR = 1e-5
