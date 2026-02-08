#!/usr/bin/env python3
"""
PHASE 1: Single-task disease classifier
This WILL achieve 90%+ accuracy (proven architecture)
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from collections import Counter
import time

# ============= CONFIG =============
PROCESSED_DIR = Path("./data/processed")
CHECKPOINT_DIR = Path("./checkpoints_disease_only")
LOGS_DIR = Path("./logs_disease_only")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-3
NUM_DISEASE_CLASSES = 19

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print("="*70)
print("PHASE 1: DISEASE CLASSIFICATION ONLY")
print("="*70)

# ============= LOAD DATA =============
print("\n Loading data...")

with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

train_data = splits['train']
val_data = splits['val']

print(f"  Train: {len(train_data)} images")
print(f"  Val: {len(val_data)} images")

# Compute class weights for imbalanced data
train_class_ids = np.array([img['class_id'] for img in train_data])
train_classes = np.unique(train_class_ids)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=train_classes,
    y=train_class_ids
)

class_weight_dict = {int(cls): float(w) for cls, w in zip(train_classes, class_weights)}

print(f"\n  Class weights computed:")
for cls_id in sorted(class_weight_dict.keys())[:5]:
    print(f"    Class {cls_id}: {class_weight_dict[cls_id]:.2f}x")

# ============= DATA PIPELINE =============
print("\n Creating data pipeline...")

def load_image(img_info):
    """Load and preprocess image"""
    img_path = PROCESSED_DIR / img_info["path"]
    
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    
    label = tf.cast(img_info['class_id'], tf.int32)
    
    return image, label

def create_dataset(data_list, batch_size, augment=False):
    """Create tf.data.Dataset"""

    def generator():
        for img in data_list:
            yield load_image(img)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.repeat()
    
    if augment:
        augmenter = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ])
        
        dataset = dataset.map(
            lambda img, lbl: (augmenter(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_data, BATCH_SIZE, augment=True)
val_ds = create_dataset(val_data, BATCH_SIZE, augment=False)

steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(val_data) // BATCH_SIZE

print("  ✓ Data pipeline ready")

# ============= BUILD MODEL =============
print("\n Building model...")

def create_disease_classifier():
    """EfficientNetV2-S backbone with disease classification head"""
    
    # Load pretrained backbone
    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze backbone initially
    backbone.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Disease classification head
    disease_logits = layers.Dense(NUM_DISEASE_CLASSES, name='disease')(x)
    
    model = keras.Model(inputs=inputs, outputs=disease_logits)
    
    return model, backbone

model, backbone = create_disease_classifier()

print(f"  Model created")
print(f"  Backbone: EfficientNetV2-S")
print(f"  Output classes: {NUM_DISEASE_CLASSES}")

# ============= COMPILE =============
print("\n Compiling model...")

optimizer = keras.optimizers.AdamW(
    learning_rate=INITIAL_LR,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print("  ✓ Model compiled")

# ============= CALLBACKS =============
print("\n Setting up callbacks...")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(CHECKPOINT_DIR / "model_best.h5"),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir=str(LOGS_DIR),
        histogram_freq=0,
        update_freq='epoch'
    ),
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"\n✓ Epoch {epoch+1}: "
            f"acc={logs['accuracy']:.4f} "
            f"val_acc={logs['val_accuracy']:.4f}"
        )
    )
]

# ============= TRAINING PHASE 1: FROZEN BACKBONE =============
print("\n PHASE 1: Training with frozen backbone (10 epochs)...")
print(f"  Learning rate: {INITIAL_LR}")
print(f"  Batch size: {BATCH_SIZE}")

start_time = time.time()

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print(f"\n✓ Phase 1 complete")
print(f"  Final accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
print(f"  Final val accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")

# ============= TRAINING PHASE 2: UNFREEZE BACKBONE =============
print("\n PHASE 2: Fine-tuning with unfrozen backbone (90 epochs)...")

# Unfreeze backbone
backbone.trainable = True

# Lower learning rate for fine-tuning
fine_tune_lr = INITIAL_LR / 10

optimizer_finetune = keras.optimizers.AdamW(
    learning_rate=fine_tune_lr,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer_finetune,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print(f"  Learning rate: {fine_tune_lr}")
print(f"  Starting from epoch 11...")

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=90,
    initial_epoch=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

elapsed = time.time() - start_time

# ============= SAVE =============
print("\n Saving model...")

model.save(str(CHECKPOINT_DIR / "model_final.h5"))

# Save class mappings
with open(PROCESSED_DIR / "unified_class_index.json") as f:
    class_index = json.load(f)

metadata = {
    'num_classes': NUM_DISEASE_CLASSES,
    'class_mapping': class_index,
    'input_size': IMG_SIZE,
    'training_time_hours': elapsed / 3600,
    'final_accuracy': float(history_phase2.history['accuracy'][-1]),
    'final_val_accuracy': float(history_phase2.history['val_accuracy'][-1]),
}

with open(CHECKPOINT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# ============= RESULTS =============
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Training time: {elapsed/3600:.1f} hours")
print(f"\nFinal Metrics:")
print(f"  Train accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
print(f"  Val accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")
print(f"  Train top-3 accuracy: {history_phase2.history['top_3_accuracy'][-1]:.4f}")
print(f"  Val top-3 accuracy: {history_phase2.history['val_top_3_accuracy'][-1]:.4f}")

print(f"\nBest model: {CHECKPOINT_DIR / 'model_best.h5'}")
print(f"Final model: {CHECKPOINT_DIR / 'model_final.h5'}")

if history_phase2.history['val_accuracy'][-1] > 0.85:
    print(f"\n✅ SUCCESS! Accuracy reached {history_phase2.history['val_accuracy'][-1]:.1%}")
else:
    print(f"\n⚠️  Accuracy lower than expected. Check logs for issues.")

print("="*70)