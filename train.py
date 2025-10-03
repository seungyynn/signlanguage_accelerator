import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # Safe to ignore if not supported
        pass


def build_data_augmentation() -> keras.Sequential:
    # Keep augmentations modest to avoid altering the sign semantics
    return keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            # Avoid horizontal flip by default for sign language
        ],
        name="data_augmentation",
    )


def create_datasets(
    data_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    # Cache and prefetch for performance
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, class_names


def build_model(
    num_classes: int,
    image_size: Tuple[int, int],
    base_model_name: str,
    dropout: float,
    learning_rate: float,
) -> Tuple[keras.Model, keras.Model]:
    height, width = image_size
    inputs = keras.Input(shape=(height, width, 3), dtype=tf.float32)

    # Normalize to [-1, 1] to match MobileNet family expectations
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale_minus1_1")(inputs)

    x = build_data_augmentation()(x)

    base_model: keras.Model
    base_model_name = base_model_name.lower()
    if base_model_name in {"mobilenetv3", "mobilenetv3small", "mnv3", "mnv3s"}:
        base_model = keras.applications.MobileNetV3Small(
            input_shape=(height, width, 3), include_top=False, weights="imagenet", pooling="avg"
        )
    elif base_model_name in {"mobilenetv2", "mnv2"}:
        base_model = keras.applications.MobileNetV2(
            input_shape=(height, width, 3), include_top=False, weights="imagenet", pooling="avg"
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    base_model.trainable = False

    x = base_model(x, training=False)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model.name}_asl_alphabet")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model, base_model


def fine_tune(
    model: keras.Model,
    base_model: keras.Model,
    unfreeze_layers: int,
    fine_tune_lr: float,
) -> None:
    if unfreeze_layers <= 0:
        return

    base_model.trainable = True
    # Freeze all but the last N layers of the base model
    if unfreeze_layers < len(base_model.layers):
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(fine_tune_lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def write_labels_file(labels: List[str], output_dir: str) -> str:
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")
    return labels_path


def convert_to_tflite_fp16(model: keras.Model, out_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def convert_to_tflite_int8(
    model: keras.Model,
    rep_ds: tf.data.Dataset,
    out_path: str,
    max_samples: int = 500,
) -> None:
    # Prepare representative dataset: unbatch to yield single images
    rep_images = rep_ds.unbatch().map(lambda x, _: x).take(max_samples).batch(1)

    def representative_dataset() -> Iterable[List[np.ndarray]]:
        for batch in rep_images:
            # Ensure float32 input (the model contains its own Rescaling layer)
            img = tf.cast(batch, tf.float32)
            yield [img]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def train_and_export(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    configure_gpu_memory_growth()

    image_size = (args.image_size, args.image_size)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {args.data_dir}")
    train_ds, val_ds, class_names = create_datasets(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    print("Building model…")
    model, base_model = build_model(
        num_classes=num_classes,
        image_size=image_size,
        base_model_name=args.base_model,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    best_model_path = os.path.join(run_dir, "best.keras")

    callbacks_warmup = [
        keras.callbacks.CSVLogger(os.path.join(run_dir, "training_log_warmup.csv")),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    print("Stage 1: Training classifier head (base frozen)…")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        callbacks=callbacks_warmup,
        verbose=1,
    )

    print("Stage 2: Fine-tuning top layers…")
    fine_tune(
        model=model,
        base_model=base_model,
        unfreeze_layers=args.unfreeze_layers,
        fine_tune_lr=args.fine_tune_lr,
    )

    callbacks_finetune = [
        keras.callbacks.ModelCheckpoint(
            best_model_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=args.early_stop_patience, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(os.path.join(run_dir, "training_log_finetune.csv")),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks_finetune,
        verbose=1,
    )

    # Ensure best weights are loaded (EarlyStopping.restore_best_weights=True already does this)
    model.save(os.path.join(run_dir, "final.keras"))

    # Persist labels for inference
    labels_path = write_labels_file(class_names, run_dir)
    print(f"Saved labels to: {labels_path}")

    # Export TFLite models
    fp16_path = os.path.join(run_dir, "model_fp16.tflite")
    print("Converting to TFLite FP16…")
    convert_to_tflite_fp16(model, fp16_path)
    print(f"Saved FP16 TFLite: {fp16_path}")

    int8_path = os.path.join(run_dir, "model_int8.tflite")
    try:
        print("Converting to TFLite INT8 (full integer)…")
        convert_to_tflite_int8(model, rep_ds=train_ds, out_path=int8_path, max_samples=args.representative_samples)
        print(f"Saved INT8 TFLite: {int8_path}")
    except Exception as e:
        print("INT8 conversion failed; you can still use FP16 model.")
        print(f"Reason: {e}")

    print("All done.")
    print(f"Artifacts in: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNet for ASL alphabet and export TFLite.")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Root directory with class subfolders A…Z")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to write models and logs")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--unfreeze-layers", type=int, default=40, help="Unfreeze last N layers during fine-tune")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--representative-samples", type=int, default=500, help="Samples for INT8 calibration")
    parser.add_argument("--base-model", type=str, default="MobileNetV3Small", help="MobileNetV2 or MobileNetV3Small")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_export(args)
