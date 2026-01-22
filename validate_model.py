#!/usr/bin/env python3
"""Final disease model validation"""

import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np

PROCESSED_DIR = Path("./data/processed")
CHECKPOINT_DIR = Path("./checkpoints_disease_only")
NUM_CLASSES = 19  # Model was trained with 19 classes (0-18)

print("=" * 70)
print("DISEASE MODEL VALIDATION")
print("=" * 70)

# Load best model
print("\n[1] Loading model...")
model = keras.models.load_model(str(CHECKPOINT_DIR / "model_best.h5"), compile=False)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"    Model output classes: {model.output_shape[-1]}")

# Load val data
print("\n[2] Loading validation data...")
with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

# Filter to only classes the model was trained on (0-18)
val_data = [img for img in splits['val'] if img['class_id'] < NUM_CLASSES]
print(f"    Total val images: {len(splits['val'])}")
print(f"    Filtered (classes 0-{NUM_CLASSES-1}): {len(val_data)}")

# Load class names for reporting
with open(PROCESSED_DIR / "unified_class_index.json") as f:
    class_index = json.load(f)

# Create validation dataset
def load_image(img_info):
    img_path = PROCESSED_DIR / img_info["path"]
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(img_info['class_id'], tf.int32)
    return image, label

print("\n[3] Creating dataset...")
dataset = tf.data.Dataset.from_generator(
    (lambda: (load_image(img) for img in val_data)),
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).batch(32).prefetch(tf.data.AUTOTUNE)

# Evaluate
print("\n[4] Evaluating...")
loss, accuracy = model.evaluate(dataset, verbose=1)

print("\n" + "=" * 70)
print(f"✅ VALIDATION ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Loss: {loss:.4f}")
print("=" * 70)

# Show which classes were validated
print("\nClasses validated:")
for i in range(NUM_CLASSES):
    class_name = class_index.get(str(i), f"Unknown_{i}")
    count = sum(1 for img in val_data if img['class_id'] == i)
    print(f"  {i:2d}: {class_name:<45} ({count} images)")
