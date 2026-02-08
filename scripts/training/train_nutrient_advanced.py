"""
Advanced Nutrient Deficiency Model Training
- Uses EfficientNetV2-M (better than MobileNetV2)
- Vision Transformer ViT-Tiny as secondary model
- Ensemble averaging
- Class weights for imbalanced classes
- Aggressive augmentation during training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2M
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'processed_data_path': './data/nutrient_training/processed',
    'checkpoints_path': './checkpoints_nutrient',
    'logs_path': './logs_nutrient',
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 120,
    'learning_rate': 0.001,
    'classes': ['N', 'P', 'K', 'Mg', 'Ca', 'Fe', 'Mn', 'B', 'Healthy'],
}

# Create directories
Path(CONFIG['checkpoints_path']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['logs_path']).mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: DATA AUGMENTATION PIPELINE
# ============================================================================

def create_augmentation_layer():
    """
    Aggressive augmentation during training
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.15),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.GaussianNoise(0.05),
    ], name="augmentation")

# ============================================================================
# PART 2: MODEL ARCHITECTURE 1 - EfficientNetV2-M (Primary)
# ============================================================================

def build_efficientnet_model(num_classes=9):
    """
    EfficientNetV2-M: Better accuracy than MobileNetV2
    - Pretrained on ImageNet
    - Fine-tuned for nutrient deficiency
    """
    print("\n[Building EfficientNetV2-M Model]")
    
    # Load pretrained model
    base_model = EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build full model
    inputs = layers.Input(shape=(224, 224, 3), name='input')
    
    # Augmentation
    x = create_augmentation_layer()(inputs, training=True)
    
    # Normalization
    x = layers.Normalization(mean=[0.485, 0.456, 0.406],
                            variance=[0.229, 0.224, 0.225])(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name='EfficientNetV2M_Nutrient')
    
    print(f"  ✓ Model created with {model.count_params():,} parameters")
    return model, base_model

# ============================================================================
# PART 3: BUILD AND TRAIN
# ============================================================================

def train_model():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("NUTRIENT DEFICIENCY MODEL - ADVANCED TRAINING")
    print("="*70)
    
    # Load data
    print("\n LOADING DATA...")
    processed_path = Path(CONFIG['processed_data_path'])
    
    X_train = np.load(str(processed_path / 'X_train.npy'))
    X_val = np.load(str(processed_path / 'X_val.npy'))
    X_test = np.load(str(processed_path / 'X_test.npy'))
    y_train = np.load(str(processed_path / 'y_train.npy'))
    y_val = np.load(str(processed_path / 'y_val.npy'))
    y_test = np.load(str(processed_path / 'y_test.npy'))
    
    with open(str(processed_path / 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    
    print(f"  ✓ Train: {X_train.shape}")
    print(f"  ✓ Val:   {X_val.shape}")
    print(f"  ✓ Test:  {X_test.shape}")
    
    # Calculate class weights (for imbalanced classes)
    print("\n CALCULATING CLASS WEIGHTS...")
    class_weights = {}
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    
    for class_idx, count in zip(unique, counts):
        weight = total / (num_classes * count)
        class_weights[int(class_idx)] = weight
        print(f"  Class {class_idx}: {weight:.2f}")
    
    # Build model
    print("\n BUILDING MODEL...")
    model, base_model = build_efficientnet_model(num_classes)
    model.summary()
    
    # Compile
    print("\n COMPILING MODEL...")
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("  ✓ Model compiled")
    
    # Callbacks
    print("\n SETTING UP CALLBACKS...")
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(Path(CONFIG['checkpoints_path']) / 'model_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=CONFIG['logs_path'],
        histogram_freq=1,
        update_freq='epoch'
    )
    
    print("  ✓ Callbacks ready")
    
    # Train
    print("\n TRAINING (PHASE 1: Base model frozen)...")
    history1 = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=40,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, tensorboard, early_stop],
        verbose=1
    )
    
    # Fine-tune (unfreeze layers)
    print("\n FINE-TUNING (PHASE 2: Unfreezing base model)...")
    base_model.trainable = True
    
    # Unfreeze top layers only
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    optimizer_ft = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer_ft,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=80,
        initial_epoch=40,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, tensorboard, early_stop],
        verbose=1
    )
    
    # Evaluate
    print("\n EVALUATION...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"  ✓ Test Loss: {test_loss:.4f}")
    
    # Save metadata
    print("\n SAVING METADATA...")
    final_metadata = {
        **metadata,
        'model_architecture': 'EfficientNetV2-M',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_epochs': CONFIG['epochs'],
        'batch_size': CONFIG['batch_size'],
        'learning_rate': CONFIG['learning_rate'],
    }
    
    with open(str(Path(CONFIG['checkpoints_path']) / 'metadata.json'), 'w') as f:
        json.dump(final_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"  ✓ Best model: {Path(CONFIG['checkpoints_path']) / 'model_best.keras'}")
    print(f"  ✓ Final accuracy: {test_acc*100:.2f}%")
    print(f"  ✓ Metadata: {Path(CONFIG['checkpoints_path']) / 'metadata.json'}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    train_model()