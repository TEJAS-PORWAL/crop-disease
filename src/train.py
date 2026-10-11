"""Training entrypoint: two-phase transfer learning + fine-tuning."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless backend so plots save without a display.
import matplotlib.pyplot as plt
import tensorflow as tf

import config
from src.data_loader import load_datasets, save_class_names
from src.model import build_model, compile_for_fine_tune, compile_for_head


def _plot_history(histories: list[tf.keras.callbacks.History]) -> None:
    """Combine phase-1 and phase-2 histories and save accuracy/loss curves."""
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, acc, "o-", label="Train")
    ax1.plot(epochs, val_acc, "o-", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs, loss, "o-", label="Train")
    ax2.plot(epochs, val_loss, "o-", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(config.HISTORY_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"Saved training history plot to {config.HISTORY_PLOT_PATH}")


def train() -> None:
    config.ensure_dirs()
    tf.random.set_seed(config.SEED)

    print("Loading datasets...")
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    save_class_names(class_names)

    print("Building model...")
    model = build_model(num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    # --- Phase 1: train the classifier head --------------------------------
    print("\n=== Phase 1: training classifier head (base frozen) ===")
    compile_for_head(model)
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.HEAD_EPOCHS,
        callbacks=[early_stop],
    )

    # --- Phase 2: fine-tune the top of the base model ----------------------
    print("\n=== Phase 2: fine-tuning top layers of base model ===")
    compile_for_fine_tune(model)
    total_epochs = config.HEAD_EPOCHS + config.FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=len(history_head.history["loss"]),
        callbacks=[early_stop],
    )

    print(f"\nSaving model to {config.MODEL_PATH}")
    model.save(config.MODEL_PATH)

    _plot_history([history_head, history_fine])

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    train()
