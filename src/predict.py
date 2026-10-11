"""Single-image prediction CLI and reusable inference helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import config
from src.data_loader import load_class_names

_MODEL_CACHE: tf.keras.Model | None = None


def load_trained_model() -> tf.keras.Model:
    """Load (and cache) the trained model from disk."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at '{config.MODEL_PATH}'. "
                "Train first with `python -m src.train`."
            )
        _MODEL_CACHE = tf.keras.models.load_model(config.MODEL_PATH)
    return _MODEL_CACHE


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load an image file and return a model-ready batch of shape (1, H, W, 3)."""
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)  # values in [0, 255]
    return np.expand_dims(arr, axis=0)


def predict_array(batch: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
    """Run inference on a preprocessed batch and return top-k (label, prob)."""
    model = load_trained_model()
    class_names = load_class_names()
    probs = model.predict(batch, verbose=0)[0]
    k = min(top_k, len(class_names))
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def predict_image(image_path: str | Path, top_k: int = 3) -> List[Tuple[str, float]]:
    """Predict the disease class for a single image file."""
    batch = preprocess_image(image_path)
    return predict_array(batch, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image.")
    parser.add_argument("--image", required=True, help="Path to the leaf image.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show.")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    results = predict_image(args.image, top_k=args.top_k)
    best_label, best_prob = results[0]
    print(f"\nPrediction for {args.image}:")
    print(f"  => {best_label}  ({best_prob * 100:.2f}%)\n")
    print("Top predictions:")
    for label, prob in results:
        print(f"  {label:<40} {prob * 100:6.2f}%")

    from src.treatments import get_treatment

    t = get_treatment(best_label)
    print("\nRecommended treatment:")
    print(f"  {t['summary']}")
    print(f"  Cause: {t['cause']}")
    print("  What to do:")
    for step in t["steps"]:
        print(f"    - {step}")


if __name__ == "__main__":
    main()
