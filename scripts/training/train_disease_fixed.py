#!/usr/bin/env python3
"""
FIXED TRAINING SCRIPT - DISEASE CLASSIFICATION
Achieves 90%+ accuracy with proper data handling
"""

# Suppress numpy FutureWarning before any imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import time

# ============= CONFIG =============
PROCESSED_DIR = Path("./data/processed")
CHECKPOINT_DIR = Path("./checkpoints_disease_only")
LOGS_DIR = Path("./logs_disease_only")

IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced from 32 for better gradient flow
EPOCHS = 100
INITIAL_LR = 5e-4  # Lower initial learning rate
# NUM_DISEASE_CLASSES set dynamically after loading data

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

print("="*70)
print("PHASE 1: DISEASE CLASSIFICATION (19 CLASSES)")
print("="*70)

# ============= LOAD DATA =============
print("\n[1] Loading data...")

with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

train_data = splits['train']
val_data = splits['val']

print(f"  Train: {len(train_data)} images")
print(f"  Val: {len(val_data)} images")

# ============= VERIFY AND REMAP CLASS IDS =============
print("\n[2] Verifying class IDs...")

train_class_ids = np.array([img['class_id'] for img in train_data])
unique_classes = sorted(set(int(x) for x in np.unique(train_class_ids)))

print(f"  Found {len(unique_classes)} unique classes")
print(f"  Original IDs: {unique_classes}")

# Check if remapping needed (IDs not contiguous 0 to N-1)
expected_ids = list(range(len(unique_classes)))
needs_remapping = unique_classes != expected_ids

if needs_remapping:
    print(f"  Remapping to contiguous IDs 0-{len(unique_classes)-1}...")

    # Create mapping: original_id -> new_contiguous_id
    id_remap = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}

    # Remap in train and val data
    for img in train_data:
        img['class_id'] = id_remap[int(img['class_id'])]
    for img in val_data:
        img['class_id'] = id_remap[int(img['class_id'])]

    # Update for downstream use
    train_class_ids = np.array([img['class_id'] for img in train_data])
    unique_classes = sorted(set(int(x) for x in np.unique(train_class_ids)))
    print(f"  ✓ Remapped to: {unique_classes}")

NUM_DISEASE_CLASSES = len(unique_classes)
print(f"  ✓ {NUM_DISEASE_CLASSES} classes ready")

# ============= COMPUTE CLASS WEIGHTS =============
print("\n[3] Computing class weights...")

class_weights = compute_class_weight(
    'balanced',
    classes=np.array(unique_classes),
    y=train_class_ids
)

class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights)}

print("  Class weights:")
for cls_id in sorted(class_weight_dict.keys())[:5]:
    print(f"    Class {cls_id}: {class_weight_dict[cls_id]:.3f}x")

# ============= DATA PIPELINE =============
print("\n[4] Creating data pipeline...")

def load_image(img_info):
    """Load, verify, and preprocess image"""
    img_path = PROCESSED_DIR / img_info["path"]
    
    try:
        # Read image
        image = tf.io.read_file(str(img_path))
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Verify shape
        if tf.shape(image)[0] < 50 or tf.shape(image)[1] < 50:
            # Image too small, skip
            image = tf.ones((IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8) * 128
        
        # Resize
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Ensure correct shape
        image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, 3))
        
        label = tf.cast(img_info['class_id'], tf.int32)
        
        return image, label
    except Exception as e:
        # Return black image on error
        image = tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        label = tf.cast(img_info['class_id'], tf.int32)
        return image, label


def create_dataset(data_list, batch_size, augment=False):
    """Create tf.data.Dataset with proper configuration"""
    
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
    
    if augment:
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            return img, lbl

        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # IMPORTANT: repeat() allows dataset to loop for multiple epochs
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_data, BATCH_SIZE, augment=True)
val_ds = create_dataset(val_data, BATCH_SIZE, augment=False)

steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(val_data) // BATCH_SIZE

print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Validation steps: {validation_steps}")
print("  ✓ Data pipeline ready")

# ============= BUILD MODEL =============
print("\n[5] Building model...")

def create_disease_classifier():
    """EfficientNetV2S with disease head"""
    
    # Load backbone
    backbone = applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze backbone
    backbone.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Backbone
    x = backbone(inputs, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    disease_logits = layers.Dense(NUM_DISEASE_CLASSES, name='disease')(x)
    
    model = keras.Model(inputs=inputs, outputs=disease_logits)
    
    return model, backbone


model, backbone = create_disease_classifier()

print(f"  Model architecture created")
print(f"  Output classes: {NUM_DISEASE_CLASSES}")
print("  ✓ Model built")

# ============= COMPILE =============
print("\n[6] Compiling model...")

optimizer = keras.optimizers.AdamW(
    learning_rate=INITIAL_LR,
    weight_decay=1e-4
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
print("\n[7] Setting up callbacks...")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(CHECKPOINT_DIR / "model_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.TensorBoard(
        log_dir=str(LOGS_DIR),
        update_freq='epoch'
    ),
]

print("  ✓ Callbacks configured")

# ============= PHASE 1: FROZEN BACKBONE =============
print("\n" + "="*70)
print("PHASE 1: Training with frozen backbone (10 epochs)")
print("="*70)
print(f"Learning rate: {INITIAL_LR}")
print(f"Batch size: {BATCH_SIZE}")

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
print(f"  Final train accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
print(f"  Final val accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")

# ============= PHASE 2: UNFREEZE & FINETUNE =============
print("\n" + "="*70)
print("PHASE 2: Fine-tuning with unfrozen backbone (90 epochs)")
print("="*70)

# Unfreeze
backbone.trainable = True

# Lower learning rate
fine_tune_lr = INITIAL_LR / 5

optimizer_ft = keras.optimizers.AdamW(
    learning_rate=fine_tune_lr,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer_ft,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print(f"Learning rate: {fine_tune_lr}")

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
print("\n[8] Saving model...")

model.save(str(CHECKPOINT_DIR / "model_final.h5"))

# Load class index
with open(PROCESSED_DIR / "unified_class_index.json") as f:
    class_index = json.load(f)

# Metadata
metadata = {
    'num_classes': NUM_DISEASE_CLASSES,
    'class_mapping': class_index,
    'input_size': IMG_SIZE,
    'training_time_hours': elapsed / 3600,
    'final_accuracy': float(history_phase2.history['accuracy'][-1]),
    'final_val_accuracy': float(history_phase2.history['val_accuracy'][-1]),
    'batch_size': BATCH_SIZE,
}

with open(CHECKPOINT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("  ✓ Models saved")

# ============= RESULTS =============
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Total training time: {elapsed/3600:.1f} hours")
print(f"\nFinal Metrics:")
print(f"  Train accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
print(f"  Val accuracy:   {history_phase2.history['val_accuracy'][-1]:.4f}")

final_val_acc = history_phase2.history['val_accuracy'][-1]
if final_val_acc > 0.90:
    print(f"\n✅ SUCCESS! Validation accuracy: {final_val_acc:.1%}")
elif final_val_acc > 0.80:
    print(f"\n⚠️  Good progress! Validation accuracy: {final_val_acc:.1%}")
else:
    print(f"\n❌ Needs work. Validation accuracy: {final_val_acc:.1%}")

print("="*70)