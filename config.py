"""Central configuration for the crop disease detection system.

All paths are resolved relative to this file so the project works regardless of
the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_DIR = DATA_DIR / "raw"  # Where the Kaggle archive is extracted.

MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# --- Dataset download --------------------------------------------------------
# Kaggle dataset slug (owner/dataset-name).
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
# Fraction of each class held out for validation.
VAL_SPLIT = 0.2

# --- Image / data ------------------------------------------------------------
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# --- Training ----------------------------------------------------------------
# Phase 1: train the classification head with the base model frozen.
HEAD_EPOCHS = 10
HEAD_LEARNING_RATE = 1e-3

# Phase 2: unfreeze the top layers of the base model and fine-tune.
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 1e-5
# Number of layers at the end of the base model to make trainable.
FINE_TUNE_AT = 100

DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 5


def ensure_dirs() -> None:
    """Create the directories the project writes to."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
