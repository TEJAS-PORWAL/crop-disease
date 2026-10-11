"""Evaluate a trained model: confusion matrix + classification report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.data_loader import load_class_names, load_datasets


def evaluate() -> None:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at '{config.MODEL_PATH}'. "
            "Train first with `python -m src.train`."
        )

    print("Loading datasets and model...")
    _, val_ds, class_names_ds = load_datasets()
    class_names = load_class_names()
    if class_names != class_names_ds:
        print(
            "Warning: saved class names differ from dataset order. "
            "Using the saved order from training."
        )

    model = tf.keras.models.load_model(config.MODEL_PATH)

    print("Running predictions on the validation set...")
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    print("\n=== Classification report ===")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")

    accuracy = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
    print(f"Overall accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
