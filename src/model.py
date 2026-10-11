"""Model definition: MobileNetV2 transfer learning for crop disease detection."""
from __future__ import annotations

import tensorflow as tf

import config
from src.data_loader import build_augmentation


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a transfer-learning classifier on top of MobileNetV2.

    The returned model expects raw images with pixel values in [0, 255];
    MobileNetV2 preprocessing is applied inside the graph so the same input
    format is used for training and inference.
    """
    inputs = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))

    # Augmentation only runs during training (training=True in fit).
    x = build_augmentation()(inputs)

    # MobileNetV2 expects inputs scaled to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base_model.trainable = False  # Phase 1: freeze the base.

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="crop_disease_classifier")
    return model


def get_base_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return the MobileNetV2 base submodel from a built model.

    Looks the layer up by name first, then falls back to locating the nested
    Functional/Model layer by type so it works across Keras versions that may
    not honor the ``name`` argument.
    """
    try:
        return model.get_layer("mobilenetv2_base")
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        raise


def compile_for_head(model: tf.keras.Model) -> None:
    """Compile the model for phase 1 (training the classifier head)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.HEAD_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_for_fine_tune(model: tf.keras.Model) -> None:
    """Unfreeze top layers of the base and compile with a low learning rate."""
    base_model = get_base_model(model)
    base_model.trainable = True

    # Freeze everything before FINE_TUNE_AT; fine-tune the rest.
    for layer in base_model.layers[: config.FINE_TUNE_AT]:
        layer.trainable = False
    # Keep BatchNorm layers frozen during fine-tuning for stability.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
