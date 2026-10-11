"""Generate a small synthetic dataset so the pipeline can run end-to-end.

This creates toy images with distinct visual signatures per class (base color,
noise, and simple shapes). It is NOT real leaf data — it exists only so you can
verify training, evaluation, prediction, and the web app without a large
download. Swap in the real PlantVillage dataset for meaningful results.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import config

# A few representative PlantVillage-style class names.
CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Late_blight",
]

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10
IMG_PX = 160


def _class_signature(class_idx: int) -> dict:
    """Deterministic visual signature per class."""
    rng = np.random.default_rng(seed=1000 + class_idx)
    base_color = rng.integers(30, 200, size=3)
    spot_color = rng.integers(0, 255, size=3)
    num_spots = int(rng.integers(3, 12))
    return {
        "base_color": base_color,
        "spot_color": spot_color,
        "num_spots": num_spots,
    }


def _make_image(class_idx: int, sample_idx: int) -> Image.Image:
    sig = _class_signature(class_idx)
    rng = np.random.default_rng(seed=class_idx * 10_000 + sample_idx)

    # Base canvas with per-class color + mild noise.
    base = np.tile(sig["base_color"].astype(np.uint8), (IMG_PX, IMG_PX, 1))
    noise = rng.integers(-20, 20, size=(IMG_PX, IMG_PX, 3))
    canvas = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw class-characteristic "lesion" spots.
    spot_color = tuple(int(c) for c in sig["spot_color"])
    for _ in range(sig["num_spots"]):
        cx = int(rng.integers(0, IMG_PX))
        cy = int(rng.integers(0, IMG_PX))
        r = int(rng.integers(4, 18))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=spot_color)

    return img


def generate() -> None:
    config.ensure_dirs()
    print("Generating synthetic dataset (toy data for pipeline testing)...")

    for split, count in (("train", TRAIN_PER_CLASS), ("val", VAL_PER_CLASS)):
        split_dir = config.DATA_DIR / split
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = _make_image(class_idx, i if split == "train" else i + 10_000)
                img.save(class_dir / f"img_{i:03d}.png")

    total = len(CLASSES) * (TRAIN_PER_CLASS + VAL_PER_CLASS)
    print(f"Done. Created {total} images across {len(CLASSES)} classes.")
    print(f"  Train: {config.TRAIN_DIR}")
    print(f"  Val:   {config.VAL_DIR}")
    print("\nReminder: this is synthetic data. Use real images for real results.")


if __name__ == "__main__":
    generate()
