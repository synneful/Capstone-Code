"""
===================================================================
Fine-Tuning Phase 2 - Unfreeze MobileNetV2 base layers
===================================================================
Purpose:
    Your first training run (20 epochs, head-only) plateaued around
    58% validation accuracy because MobileNetV2's base layers were
    frozen - only the small classification head was learning.

    This script picks up from that already-trained model, unfreezes
    the LAST N layers of the base network, and continues training
    at a much lower learning rate. This lets those layers adapt
    specifically to food images instead of staying generic ImageNet
    features - this is the step that typically gives the real
    accuracy jump in transfer learning.

Before running:
    Upload your previous run's 'best_model.keras' (from model_output/)
    alongside this script, and have food-101/images and raw_food_dataset
    available exactly as before.

Usage:
    python finetune_classifier.py \
        --previous_model ./best_model.keras \
        --food101_dir ./food-101/images \
        --raw_data_dir ./raw_food_dataset \
        --epochs 10 \
        --unfreeze_layers 30
===================================================================
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


def load_split_from_single_dir(directory, subset, validation_split=0.2):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True
    )
    class_names = ds.class_names
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds, class_names


def load_split_from_separate_dirs(base_dir, split_folder_name):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, split_folder_name),
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True
    )
    class_names = ds.class_names
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds, class_names


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous_model", type=str, required=True,
                         help="Path to best_model.keras from the first training run")
    parser.add_argument("--food101_dir", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--unfreeze_layers", type=int, default=30,
                         help="Number of layers (from the end) of MobileNetV2 to unfreeze")
    parser.add_argument("--output_dir", type=str, default="./model_output_finetuned")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load datasets exactly as before (same seed = same split) ---
    train_ds_dishes, dish_class_names = load_split_from_single_dir(args.food101_dir, "training")
    val_ds_dishes, _ = load_split_from_single_dir(args.food101_dir, "validation")
    num_dish_classes = len(dish_class_names)

    train_ds_raw, raw_class_names = load_split_from_separate_dirs(args.raw_data_dir, "train")
    val_ds_raw, _ = load_split_from_separate_dirs(args.raw_data_dir, "validation")

    train_ds_raw = train_ds_raw.map(lambda x, y: (x, y + num_dish_classes))
    val_ds_raw = val_ds_raw.map(lambda x, y: (x, y + num_dish_classes))

    combined_class_names = list(dish_class_names) + list(raw_class_names)
    with open(os.path.join(args.output_dir, "class_indices.json"), "w") as f:
        json.dump({i: name for i, name in enumerate(combined_class_names)}, f, indent=2)

    train_ds_dishes = train_ds_dishes.map(normalize).map(augment)
    train_ds_raw = train_ds_raw.map(normalize).map(augment)
    val_ds_dishes = val_ds_dishes.map(normalize)
    val_ds_raw = val_ds_raw.map(normalize)

    train_ds = train_ds_dishes.concatenate(train_ds_raw)
    val_ds = val_ds_dishes.concatenate(val_ds_raw)

    train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Load the already-trained model ---
    model = load_model(args.previous_model)

    # NOTE: because the original model was built as
    # Model(inputs=base_model.input, outputs=...), Keras flattens
    # MobileNetV2's internal layers directly into model.layers instead of
    # keeping it as one nested sub-model. So there is no single "base_model"
    # object to call .layers on - we work with model.layers directly instead.
    #
    # The custom head we added in the original script was exactly:
    #   GlobalAveragePooling2D, Dense(256), Dropout, Dense(num_classes)
    # = 4 layers. Everything before that is MobileNetV2's base.
    head_layer_count = 4
    base_layers = model.layers[:-head_layer_count]
    head_layers = model.layers[-head_layer_count:]

    # Freeze all base layers first, then unfreeze just the last N
    for layer in base_layers:
        layer.trainable = False
    for layer in base_layers[-args.unfreeze_layers:]:
        layer.trainable = True
    # Head layers always stay trainable
    for layer in head_layers:
        layer.trainable = True

    print(f"Unfroze the last {args.unfreeze_layers} of {len(base_layers)} base layers for fine-tuning "
          f"(plus the {head_layer_count} head layers, always trainable).")

    # Much lower learning rate - critical when fine-tuning pretrained weights,
    # otherwise you destroy the useful features they already learned.
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.output_dir, "best_model_finetuned.keras"), save_best_only=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    saved_model_path = os.path.join(args.output_dir, "saved_model")
    model.export(saved_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "food_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Fine-tuned TFLite model saved to {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
