#!/usr/bin/env python3
"""
NUTRIENT DEFICIENCY TRAINING - V3 (TRANSFER LEARNING FIXED)

CRITICAL CHANGES FROM V2:
✓ Uses MobileNetV3-Small (lighter, proven to work)
✓ Starts with ImageNet pretraining (crucial!)
✓ Proper transfer learning phases
✓ Focal loss for class imbalance
✓ Ensemble-ready output
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️  GPU error: {e}")
else:
    print("⚠️  No GPU found, using CPU (slower)")

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 9,
    'batch_size': 32,
    'phase1_epochs': 20,      # Frozen base
    'phase2_epochs': 80,      # Unfrozen base
    'phase1_lr': 0.001,
    'phase2_lr': 0.0001,
    'patience': 20,
    'data_dir': './data/nutrient_training/processed',
    'checkpoint_dir': './checkpoints_nutrient',
    'log_dir': './logs_nutrient',
    'model_name': 'nutrient_v3_mobilenet'
}

# Create directories
Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("NUTRIENT DEFICIENCY TRAINING - V3 (TRANSFER LEARNING)")
print("="*80)

print("\n[CONFIG]")
for k, v in CONFIG.items():
    if k not in ['data_dir', 'checkpoint_dir', 'log_dir']:
        print(f"  {k}: {v}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[LOADING DATA]")
data_dir = Path(CONFIG['data_dir'])

X_train = np.load(data_dir / "X_train.npy").astype('float32')
y_train = np.load(data_dir / "y_train.npy").astype('int32')
X_val = np.load(data_dir / "X_val.npy").astype('float32')
y_val = np.load(data_dir / "y_val.npy").astype('int32')
X_test = np.load(data_dir / "X_test.npy").astype('float32')
y_test = np.load(data_dir / "y_test.npy").astype('int32')

print(f"✓ X_train: {X_train.shape}")
print(f"✓ X_val:   {X_val.shape}")
print(f"✓ X_test:  {X_test.shape}")

# Normalize to 0-1 if needed
if X_train.max() > 1.5:
    print("  Normalizing images from 0-255 to 0-1...")
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

print("\n[DATA AUGMENTATION PIPELINE]")

train_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="train_augmentation")

print("✓ Augmentation layers:")
print("  - Random horizontal flip")
print("  - Random rotation ±15°")
print("  - Random zoom ±15%")
print("  - Random brightness ±15%")
print("  - Random contrast ±15%")

# ============================================================================
# CLASS WEIGHTS (for imbalanced data)
# ============================================================================

print("\n[COMPUTING CLASS WEIGHTS]")

unique_classes, class_counts = np.unique(y_train, return_counts=True)
class_weights = {}

total_samples = len(y_train)
for cls, count in zip(unique_classes, class_counts):
    # Weight = inverse of class frequency
    weight = (total_samples / (len(unique_classes) * count)) ** 0.5
    class_weights[int(cls)] = float(weight)
    print(f"  Class {cls}: weight = {weight:.4f} (samples: {count})")

# ============================================================================
# BUILD MODEL
# ============================================================================

print("\n[BUILDING MODEL]")

# Load MobileNetV3-Small with ImageNet weights
base_model = MobileNetV3Small(
    input_shape=CONFIG['input_shape'],
    include_top=False,
    weights='imagenet',
    classifier_activation=None
)

print(f"✓ Base model: MobileNetV3-Small (ImageNet pretrained)")
print(f"  Trainable params: {base_model.count_params():,}")

# Build full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(CONFIG['num_classes'], activation='softmax')
], name='nutrient_classifier_v3')

print(f"✓ Full model built")
print(f"  Total parameters: {model.count_params():,}")

# ============================================================================
# CUSTOM FOCAL LOSS (better for imbalanced data)
# ============================================================================

class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), CONFIG['num_classes'])
        
        # Focal loss formula
        ce_loss = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

# ============================================================================
# PHASE 1: TRAIN TOP LAYERS WITH FROZEN BASE
# ============================================================================

print("\n[PHASE 1: TRAINING WITH FROZEN BASE]")
print(f"Epochs: {CONFIG['phase1_epochs']}")
print(f"Learning rate: {CONFIG['phase1_lr']}")
print("Base model: FROZEN")

# Freeze base
base_model.trainable = False

# Compile
optimizer_phase1 = optimizers.Adam(learning_rate=CONFIG['phase1_lr'])
model.compile(
    optimizer=optimizer_phase1,
    loss=FocalLoss(),
    metrics=['accuracy']
)

# Callbacks
checkpoint_phase1 = callbacks.ModelCheckpoint(
    filepath=f"{CONFIG['checkpoint_dir']}/model_phase1_best.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop_phase1 = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=CONFIG['patience'],
    restore_best_weights=True,
    verbose=1
)

reduce_lr_phase1 = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

tensorboard_phase1 = callbacks.TensorBoard(
    log_dir=CONFIG['log_dir'] + '/phase1',
    histogram_freq=0
)

# Train phase 1
history_phase1 = model.fit(
    X_train, y_train,
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['phase1_epochs'],
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[
        checkpoint_phase1,
        early_stop_phase1,
        reduce_lr_phase1,
        tensorboard_phase1
    ],
    verbose=1
)

# Load best phase1 model
best_phase1_acc = max(history_phase1.history['val_accuracy'])
print(f"\n✓ Phase 1 complete")
print(f"  Best validation accuracy: {best_phase1_acc:.4f}")

# ============================================================================
# PHASE 2: UNFREEZE AND FINE-TUNE
# ============================================================================

print("\n[PHASE 2: UNFREEZING BASE FOR FINE-TUNING]")

# Unfreeze last 50 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

print(f"✓ Base model unfrozen (last 50 layers trainable)")
print(f"  Trainable base layers: {sum(1 for l in base_model.layers[-50:] if l.trainable)}")

# Recompile with lower LR
optimizer_phase2 = optimizers.Adam(learning_rate=CONFIG['phase2_lr'])
model.compile(
    optimizer=optimizer_phase2,
    loss=FocalLoss(),
    metrics=['accuracy']
)

print(f"✓ Recompiled with lower learning rate: {CONFIG['phase2_lr']}")

# Callbacks for phase 2
checkpoint_phase2 = callbacks.ModelCheckpoint(
    filepath=f"{CONFIG['checkpoint_dir']}/model_best.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop_phase2 = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=CONFIG['patience'],
    restore_best_weights=True,
    verbose=1
)

reduce_lr_phase2 = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

tensorboard_phase2 = callbacks.TensorBoard(
    log_dir=CONFIG['log_dir'] + '/phase2',
    histogram_freq=0
)

# Train phase 2
print(f"Epochs: {CONFIG['phase2_epochs']}")
print(f"Learning rate: {CONFIG['phase2_lr']}")

history_phase2 = model.fit(
    X_train, y_train,
    batch_size=CONFIG['batch_size'],
    epochs=CONFIG['phase2_epochs'],
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[
        checkpoint_phase2,
        early_stop_phase2,
        reduce_lr_phase2,
        tensorboard_phase2
    ],
    verbose=1
)

best_phase2_acc = max(history_phase2.history['val_accuracy'])
print(f"\n✓ Phase 2 complete")
print(f"  Best validation accuracy: {best_phase2_acc:.4f}")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n[EVALUATION]")

# Load best model
best_model = keras.models.load_model(
    f"{CONFIG['checkpoint_dir']}/model_best.keras",
    custom_objects={'FocalLoss': FocalLoss}
)

# Test evaluation
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}")

# Predictions
y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)

# Per-class accuracy
print("\n[PER-CLASS ACCURACY]")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=4))

# ============================================================================
# SAVE METADATA
# ============================================================================

print("\n[SAVING METADATA]")

metadata = {
    'model_name': CONFIG['model_name'],
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'phase1_best_val_acc': float(best_phase1_acc),
    'phase2_best_val_acc': float(best_phase2_acc),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'total_params': int(model.count_params()),
    'base_model': 'MobileNetV3Small (ImageNet)',
    'loss_function': 'FocalLoss',
}

metadata_path = f"{CONFIG['checkpoint_dir']}/metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to {metadata_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

print(f"\nResults:")
print(f"  Phase 1 best val accuracy: {best_phase1_acc:.4f}")
print(f"  Phase 2 best val accuracy: {best_phase2_acc:.4f}")
print(f"  Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if test_accuracy >= 0.85:
    print(f"\n✅ EXCELLENT - Model ready for production!")
elif test_accuracy >= 0.75:
    print(f"\n⚠️  GOOD - Consider more data or fine-tuning")
elif test_accuracy >= 0.65:
    print(f"\n⚠️  MODERATE - Needs improvement, check data quality")
else:
    print(f"\n❌ POOR - Check data labels and distribution")

print(f"\nModel saved:")
print(f"  {CONFIG['checkpoint_dir']}/model_best.keras")
print(f"  {metadata_path}")

print("\nNext steps:")
print("  1. Review per-class accuracy above")
print("  2. If low accuracy on specific classes, check data quality")
print("  3. For production: Use ensemble of multiple models")
print("  4. Monitor: cat checkpoints_nutrient/metadata.json")

print("\n" + "="*80 + "\n")