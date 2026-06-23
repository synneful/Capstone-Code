"""
===================================================================
Combined Food Classifier - Prepared Dishes (Food-101) + Raw Produce
===================================================================
Purpose:
    Train a single classifier that recognizes BOTH:
      - 101 prepared dish categories (Food-101, downloaded directly
        from the original source - NOT via tensorflow_datasets, to
        avoid protobuf version conflicts in Colab)
      - Raw fruits/vegetables (local folder dataset, e.g. Kaggle's
        "Fruits and Vegetables Image Recognition Dataset" by Kritik Seth)

    Output is one combined label space (101 + 36 = 137 classes).

Before running, download BOTH datasets as plain folders:

    Food-101 (downloads ~5GB, already organized by class):
        !wget -q http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
        !tar -xzf food-101.tar.gz
    Expected: food-101/images/<dish_name>/*.jpg  (101 folders)

    Raw produce (from Kaggle, already has train/validation folders):
        !kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
        !unzip -q fruit-and-vegetable-image-recognition.zip -d raw_food_dataset
    Expected: raw_food_dataset/train/<food_name>/*.jpg

Usage:
    python train_combined_classifier.py \
        --food101_dir ./food-101/images \
        --raw_data_dir ./raw_food_dataset \
        --epochs 15 \
        --output_dir ./model_output
===================================================================
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


def load_split_from_single_dir(directory, subset, validation_split=0.2):
    """For datasets like Food-101 that ship as one folder per class with
    no separate train/val split - Keras carves out a consistent split
    using the same seed for both calls."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True
    )


def load_split_from_separate_dirs(base_dir, split_folder_name):
    """For datasets that already ship with separate train/ and validation/ folders."""
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, split_folder_name),
        image_size=IMG_SIZE,
        batch_size=None,
        shuffle=True
    )


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def convert_to_tflite(saved_model_path, output_path, quantize=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path} ({len(tflite_model) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--food101_dir", type=str, required=True,
                         help="Path to food-101/images (101 class folders)")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                         help="Path to the raw produce dataset (must contain train/ and validation/)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="./model_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Food-101 (single folder, carve out train/val split) ---
    train_ds_dishes = load_split_from_single_dir(args.food101_dir, "training")
    val_ds_dishes = load_split_from_single_dir(args.food101_dir, "validation")
    dish_class_names = train_ds_dishes.class_names
    num_dish_classes = len(dish_class_names)

    # --- Load raw produce dataset (already split into folders) ---
    train_ds_raw = load_split_from_separate_dirs(args.raw_data_dir, "train")
    val_ds_raw = load_split_from_separate_dirs(args.raw_data_dir, "validation")
    raw_class_names = train_ds_raw.class_names

    # --- Offset raw food labels so they sit after the dish labels ---
    train_ds_raw = train_ds_raw.map(lambda x, y: (x, y + num_dish_classes))
    val_ds_raw = val_ds_raw.map(lambda x, y: (x, y + num_dish_classes))

    combined_class_names = list(dish_class_names) + list(raw_class_names)
    num_classes = len(combined_class_names)
    print(f"Combined label space: {num_dish_classes} dishes + {len(raw_class_names)} raw foods "
          f"= {num_classes} total classes")

    with open(os.path.join(args.output_dir, "class_indices.json"), "w") as f:
        json.dump({i: name for i, name in enumerate(combined_class_names)}, f, indent=2)

    # --- Normalize + augment, then merge into one training stream ---
    train_ds_dishes = train_ds_dishes.map(normalize).map(augment)
    train_ds_raw = train_ds_raw.map(normalize).map(augment)
    val_ds_dishes = val_ds_dishes.map(normalize)
    val_ds_raw = val_ds_raw.map(normalize)

    train_ds = train_ds_dishes.concatenate(train_ds_raw)
    val_ds = val_ds_dishes.concatenate(val_ds_raw)

    train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.output_dir, "best_model.keras"), save_best_only=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    saved_model_path = os.path.join(args.output_dir, "saved_model")
    model.export(saved_model_path)

    tflite_path = os.path.join(args.output_dir, "food_classifier.tflite")
    convert_to_tflite(saved_model_path, tflite_path, quantize=True)

    print("Training and conversion complete.")
    print(f"Deploy '{tflite_path}' to your mobile app's assets folder.")
    print(f"Use '{os.path.join(args.output_dir, 'class_indices.json')}' to map prediction index -> food/dish name.")


if __name__ == "__main__":
    main()
