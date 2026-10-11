"""Download the PlantVillage dataset from Kaggle and split it into train/val.

The PlantVillage archive (``abdallahalidev/plantvillage-dataset``) contains a
top-level ``plantvillage dataset`` folder with ``color``, ``grayscale`` and
``segmented`` variants. The 38 real class folders (e.g. ``Tomato___healthy``)
live under ``color``. This script locates the color variant, then copies each
class into ``data/train`` and ``data/val`` using a per-class split so every
class is represented in both sets.

Prerequisites
-------------
1. A free Kaggle account.
2. A Kaggle API token. In Kaggle: Account > Settings > "Create New Token".
   This downloads ``kaggle.json``. Place it at:
       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
       macOS/Linux: ~/.kaggle/kaggle.json
   (Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.)

Usage
-----
    python -m src.download_data                     # all 38 classes
    python -m src.download_data --classes Tomato___healthy Tomato___Early_blight
    python -m src.download_data --crop Tomato       # only Tomato___* classes
    python -m src.download_data --max-per-class 500 # cap images per class
    python -m src.download_data --skip-download     # reuse an existing download
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import config


def _ensure_kaggle_credentials() -> None:
    """Fail early with a helpful message if Kaggle creds are missing."""
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        return
    raise SystemExit(
        "Kaggle API credentials not found.\n"
        "  1. Create a token at https://www.kaggle.com/settings (Create New Token).\n"
        f"  2. Save the downloaded kaggle.json to: {token}\n"
        "     (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables).\n"
        "Then re-run this command."
    )


def download_dataset() -> None:
    """Download and unzip the Kaggle dataset into config.RAW_DATA_DIR."""
    _ensure_kaggle_credentials()
    # Imported here so missing creds surface a clean message first.
    from kaggle.api.kaggle_api_extended import KaggleApi

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading '{config.KAGGLE_DATASET}' to {config.RAW_DATA_DIR} ...")
    print("(This is a large dataset and may take a while.)")
    api.dataset_download_files(
        config.KAGGLE_DATASET,
        path=str(config.RAW_DATA_DIR),
        unzip=True,
        quiet=False,
    )
    print("Download complete.")


def _find_color_dir() -> Path:
    """Locate the 'color' image directory inside the extracted archive.

    Falls back to the directory that contains the most class-like subfolders
    (folders whose names contain '___') if a 'color' folder isn't found.
    """
    color_dirs = [p for p in config.RAW_DATA_DIR.rglob("color") if p.is_dir()]
    if color_dirs:
        # Prefer the one that actually holds class folders.
        for d in color_dirs:
            if any(child.is_dir() and "___" in child.name for child in d.iterdir()):
                return d
        return color_dirs[0]

    # Fallback: scan for the folder richest in class-style subfolders.
    best: Optional[Path] = None
    best_count = 0
    for p in config.RAW_DATA_DIR.rglob("*"):
        if not p.is_dir():
            continue
        count = sum(1 for c in p.iterdir() if c.is_dir() and "___" in c.name)
        if count > best_count:
            best, best_count = p, count
    if best is None or best_count == 0:
        raise SystemExit(
            f"Could not find class folders under {config.RAW_DATA_DIR}. "
            "Was the dataset extracted correctly?"
        )
    return best


def _select_classes(
    source_dir: Path,
    classes: Optional[List[str]],
    crop: Optional[str],
) -> List[Path]:
    """Return the class folders to use, applying --classes / --crop filters."""
    all_classes = sorted(
        p for p in source_dir.iterdir() if p.is_dir() and "___" in p.name
    )
    if not all_classes:
        raise SystemExit(f"No class folders found in {source_dir}.")

    selected = all_classes
    if classes:
        wanted = set(classes)
        selected = [p for p in all_classes if p.name in wanted]
        missing = wanted - {p.name for p in selected}
        if missing:
            print(f"Warning: requested classes not found: {sorted(missing)}")
    elif crop:
        prefix = f"{crop}___"
        selected = [p for p in all_classes if p.name.startswith(prefix)]

    if not selected:
        available = ", ".join(p.name for p in all_classes)
        raise SystemExit(f"No classes selected. Available classes:\n{available}")
    return selected


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_into_train_val(
    classes: Optional[List[str]] = None,
    crop: Optional[str] = None,
    max_per_class: Optional[int] = None,
    val_split: float = config.VAL_SPLIT,
    clean: bool = True,
    seed: int = config.SEED,
) -> None:
    """Copy real images into data/train and data/val with a per-class split."""
    source_dir = _find_color_dir()
    print(f"Using image source: {source_dir}")

    class_dirs = _select_classes(source_dir, classes, crop)
    print(f"Selected {len(class_dirs)} class(es).")

    if clean:
        for split_dir in (config.TRAIN_DIR, config.VAL_DIR):
            if split_dir.exists():
                print(f"Clearing existing {split_dir} ...")
                shutil.rmtree(split_dir)

    rng = random.Random(seed)
    total_train = total_val = 0

    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not images:
            print(f"  Skipping '{class_dir.name}' (no images).")
            continue

        rng.shuffle(images)
        if max_per_class is not None:
            images = images[:max_per_class]

        n_val = max(1, int(len(images) * val_split)) if len(images) > 1 else 0
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, split_imgs in (("train", train_images), ("val", val_images)):
            dest = (config.TRAIN_DIR if split_name == "train" else config.VAL_DIR) / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        total_train += len(train_images)
        total_val += len(val_images)
        print(f"  {class_dir.name}: {len(train_images)} train / {len(val_images)} val")

    print(
        f"\nDone. {total_train} training and {total_val} validation images "
        f"across {len(class_dirs)} classes."
    )
    print(f"  Train: {config.TRAIN_DIR}")
    print(f"  Val:   {config.VAL_DIR}")
    print("\nNext: python -m src.train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the PlantVillage dataset from Kaggle and split it."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; reuse an already-extracted dataset in data/raw.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific class folder names to include (e.g. Tomato___healthy).",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Only include classes for this crop (e.g. Tomato, Potato).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap the number of images used per class (useful for quick tests).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=config.VAL_SPLIT,
        help=f"Validation fraction per class (default {config.VAL_SPLIT}).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing images in data/train and data/val instead of clearing.",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if not args.skip_download:
        download_dataset()
    else:
        if not config.RAW_DATA_DIR.exists():
            raise SystemExit(
                f"--skip-download given but {config.RAW_DATA_DIR} does not exist. "
                "Run without --skip-download first."
            )
        print(f"Skipping download; reusing {config.RAW_DATA_DIR}")

    split_into_train_val(
        classes=args.classes,
        crop=args.crop,
        max_per_class=args.max_per_class,
        val_split=args.val_split,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    sys.exit(main())
