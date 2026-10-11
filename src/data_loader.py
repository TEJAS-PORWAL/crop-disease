"""Dataset loading and augmentation pipelines.

Uses tf.keras.utils.image_dataset_from_directory to build tf.data pipelines
from an ImageFolder-style directory layout (one subfolder per class).
"""
from __future__ import annotations

import json
from typing import Tuple

import tensorflow as tf

import config


def _check_dir(path) -> None:
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"Dataset directory '{path}' is missing or empty. "
            "Run `python -m src.generate_synthetic_data` to create a toy "
            "dataset, or place real class folders there."
        )


def load_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load the train and validation datasets.

    Returns a tuple of (train_ds, val_ds, class_names). The datasets are
    prefetched for performance. Pixel values are kept in the [0, 255] range;
    the model handles preprocessing internally so the same pipeline works for
    training and inference.
    """
    _check_dir(config.TRAIN_DIR)
    _check_dir(config.VAL_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="int",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        seed=config.SEED,
    )

    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_augmentation() -> tf.keras.Sequential:
    """Return a data-augmentation Sequential block for leaf images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=config.SEED),
            tf.keras.layers.RandomRotation(0.2, seed=config.SEED),
            tf.keras.layers.RandomZoom(0.2, seed=config.SEED),
            tf.keras.layers.RandomContrast(0.2, seed=config.SEED),
        ],
        name="data_augmentation",
    )


def save_class_names(class_names: list[str]) -> None:
    """Persist class names so prediction stays aligned with training order."""
    config.ensure_dirs()
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


def load_class_names() -> list[str]:
    """Load class names saved during training."""
    if not config.CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found at '{config.CLASS_NAMES_PATH}'. "
            "Train the model first with `python -m src.train`."
        )
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
