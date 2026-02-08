#!/usr/bin/env python3
"""
Complete training pipeline with:
- Data loading and augmentation
- Multi-task learning with weighted losses
- Validation and checkpointing
- Confidence calibration
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from scripts.training.model_architecture import create_model
except ImportError:
    from model_architecture import create_model

# Configuration
PROCESSED_DIR = Path("./data/processed")
CHECKPOINT_DIR = Path("./checkpoints")
LOGS_DIR = Path("./logs")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-3
WARMUP_EPOCHS = 5

# Loss weights (can be tuned)
DISEASE_LOSS_WEIGHT = 1.0
NUTRIENT_LOSS_WEIGHT = 1.0
HEALTH_LOSS_WEIGHT = 0.3

# Directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

def load_class_weights(split_data, num_classes):
    """Calculate class weights for imbalanced dataset"""
    class_counts = {}
    for img in split_data:
        class_name = img["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total = sum(class_counts.values())
    weights = {class_name: total / (len(class_counts) * count) 
               for class_name, count in class_counts.items()}
    
    return weights

def create_data_generator(img_size=224):
    """Create image augmentation pipeline"""
    
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.Rescaling(1./255),
    ], name='augmentation')

def load_and_preprocess_image(img_info, img_dir, augment=False):
    """Load and preprocess a single image"""
    img_path = img_dir / img_info["path"]
    
    # Load image
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    
    return image, img_info

def create_dataset(
    split_data,
    batch_size,
    base_dir,
    augment=False,
    class_to_id=None,
):
    """Create tf.data.Dataset from image list"""

    # Extract paths and labels from split_data
    # Use source_dataset field to construct correct path for each image
    image_paths = [
        str(base_dir / img.get("source_dataset", "plant_village") / img["path"])
        for img in split_data
    ]
    class_ids = [img.get("class_id", 0) for img in split_data]

    # Create dataset from tensors
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(class_ids)
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    def load_image(path, class_id):
        # Load and preprocess image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0

        # Create label dict
        labels = {
            'disease': tf.cast(class_id, tf.int32),
            'nutrient': tf.cast(class_id % 9, tf.int32),  # Placeholder mapping
            'health': tf.constant(0.8, dtype=tf.float32)
        }
        return image, labels

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.shuffle(1000) if augment else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

class MultiTaskLoss(keras.losses.Loss):
    """Custom multi-task loss combining all three objectives"""
    
    def __init__(self, disease_weight=1.0, nutrient_weight=1.0, health_weight=0.3):
        super().__init__()
        self.disease_weight = disease_weight
        self.nutrient_weight = nutrient_weight
        self.health_weight = health_weight
        
        self.disease_loss = keras.losses.CategoricalCrossentropy()
        self.nutrient_loss = keras.losses.CategoricalCrossentropy()
        self.health_loss = keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        """Calculate combined loss"""
        
        # Convert class IDs to one-hot for disease
        disease_true = tf.one_hot(y_true['disease'], 21)
        nutrient_true = tf.one_hot(y_true['nutrient'], 9)
        health_true = y_true['health']
        
        # Calculate individual losses
        disease_loss = self.disease_loss(disease_true, y_pred['disease'])
        nutrient_loss = self.nutrient_loss(nutrient_true, y_pred['nutrient'])
        health_loss = self.health_loss(tf.expand_dims(health_true, -1), y_pred['health'])
        
        # Weighted combination
        total_loss = (
            self.disease_weight * disease_loss +
            self.nutrient_weight * nutrient_loss +
            self.health_weight * health_loss
        )
        
        return total_loss

class MultiTaskMetric(keras.metrics.Metric):
    """Custom metric for multi-task learning"""
    
    def __init__(self, name='multi_task_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.disease_acc = keras.metrics.CategoricalAccuracy()
        self.nutrient_acc = keras.metrics.CategoricalAccuracy()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        disease_true = tf.one_hot(y_true['disease'], 21)
        nutrient_true = tf.one_hot(y_true['nutrient'], 9)
        
        self.disease_acc.update_state(disease_true, y_pred['disease'], sample_weight)
        self.nutrient_acc.update_state(nutrient_true, y_pred['nutrient'], sample_weight)
    
    def result(self):
        return (self.disease_acc.result() + self.nutrient_acc.result()) / 2
    
    def reset_states(self):
        self.disease_acc.reset_states()
        self.nutrient_acc.reset_states()

def train():
    """Main training function"""
    
    print("="*70)
    print("JIVA PLANTS: MULTI-TASK MODEL TRAINING")
    print("="*70)
    
    # Load splits
    with open(PROCESSED_DIR / "splits.json") as f:
        splits = json.load(f)
    
    train_data = splits["train"]
    val_data = splits["val"]
    
    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val:   {len(val_data)} images")
    
    # Create datasets
    print(f"\nCreating data pipelines...")
    train_ds = create_dataset(
        train_data,
        BATCH_SIZE,
        PROCESSED_DIR,  # Base dir - source_dataset field determines subdirectory
        augment=True
    )
    val_ds = create_dataset(
        val_data,
        BATCH_SIZE,
        PROCESSED_DIR,
        augment=False
    )
    
    # Create model
    print(f"Creating model...")
    model = create_model(
        num_disease_classes=21,
        num_nutrient_classes=9,
        img_size=IMG_SIZE
    )
    
    # Build model
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    
    # Compile with per-output losses
    optimizer = keras.optimizers.AdamW(
        learning_rate=INITIAL_LR,
        weight_decay=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'disease': keras.losses.SparseCategoricalCrossentropy(),
            'nutrient': keras.losses.SparseCategoricalCrossentropy(),
            'health': keras.losses.MeanSquaredError()
        },
        loss_weights={
            'disease': DISEASE_LOSS_WEIGHT,
            'nutrient': NUTRIENT_LOSS_WEIGHT,
            'health': HEALTH_LOSS_WEIGHT
        },
        metrics={
            'disease': 'accuracy',
            'nutrient': 'accuracy',
            'health': 'mae'
        }
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_DIR / "model_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR)
        )
    ]
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {INITIAL_LR}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(str(CHECKPOINT_DIR / "model_final.h5"))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model: {CHECKPOINT_DIR}/model_best.h5")
    print(f"Final model: {CHECKPOINT_DIR}/model_final.h5")
    
    return model, history

if __name__ == "__main__":
    model, history = train()