#!/usr/bin/env python3
"""
FIXED training pipeline with proper label mapping
Copy from here to replace train_model.py
"""

# Suppress warnings before importing tensorflow
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import time

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

DISEASE_LOSS_WEIGHT = 1.0
NUTRIENT_LOSS_WEIGHT = 1.0
HEALTH_LOSS_WEIGHT = 0.3

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

def load_class_mappings():
    """Load unified class index and create mappings"""
    with open(PROCESSED_DIR / "unified_class_index.json") as f:
        unified_classes = json.load(f)
    
    # Map class names to IDs (0-64 for 65 classes)
    class_name_to_id = {v: int(k) for k, v in unified_classes.items()}
    
    print(f"✓ Loaded {len(class_name_to_id)} disease classes")
    
    # Nutrient classes (fixed 9 classes)
    nutrient_classes = [
        "healthy",
        "nitrogen_deficiency",
        "phosphorus_deficiency",
        "potassium_deficiency",
        "magnesium_deficiency",
        "calcium_deficiency",
        "iron_deficiency",
        "manganese_deficiency",
        "boron_deficiency"
    ]
    
    nutrient_name_to_id = {name: idx for idx, name in enumerate(nutrient_classes)}
    
    print(f"✓ Loaded {len(nutrient_classes)} nutrient classes")
    
    return class_name_to_id, nutrient_name_to_id, unified_classes

def infer_nutrient_from_disease(class_name):
    """
    Infer nutrient status from disease class name
    If no nutrient issue detected, return 'healthy' (index 0)
    """
    
    nutrient_keywords = {
        'nitrogen': 1,
        'n_deficiency': 1,
        'phosphorus': 2,
        'p_deficiency': 2,
        'potassium': 3,
        'k_deficiency': 3,
        'magnesium': 4,
        'mg_deficiency': 4,
        'calcium': 5,
        'ca_deficiency': 5,
        'iron': 6,
        'fe_deficiency': 6,
        'manganese': 7,
        'mn_deficiency': 7,
        'boron': 8,
        'b_deficiency': 8,
    }
    
    class_lower = class_name.lower()
    for keyword, nutrient_id in nutrient_keywords.items():
        if keyword in class_lower:
            return nutrient_id
    
    # Default to healthy if no nutrient deficiency detected
    return 0

def create_dataset(
    split_data,
    batch_size,
    img_dir,
    augment=False,
    class_name_to_id=None,
    nutrient_name_to_id=None,
):
    """Create tf.data.Dataset with CORRECT label mapping"""

    # Pre-compute all paths and labels (Python-side)
    img_dir_str = str(img_dir)
    image_paths = []
    disease_ids = []
    nutrient_ids = []
    health_scores = []

    for img_info in split_data:
        # Build full path
        img_path = img_info['path']
        source_dataset = img_info.get("source_dataset", "plant_village")

        # Avoid duplicating source_dataset if path already contains it
        if img_path.startswith(source_dataset):
            path = f"{img_dir_str}/{img_path}"
        else:
            path = f"{img_dir_str}/{source_dataset}/{img_path}"
        image_paths.append(path)

        # Get class name
        class_name = img_info["class"]

        # Map to disease class ID
        disease_id = class_name_to_id.get(class_name, 0)
        disease_ids.append(disease_id)

        # Infer nutrient status from disease class
        nutrient_id = infer_nutrient_from_disease(class_name)
        nutrient_ids.append(nutrient_id)

        # Health score: 0.9 if healthy, 0.5 if diseased
        is_healthy = "healthy" in class_name.lower()
        health_score = 0.9 if is_healthy else 0.5
        health_scores.append(health_score)

    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((
        image_paths,
        disease_ids,
        nutrient_ids,
        health_scores
    ))

    def load_image(path, disease_id, nutrient_id, health_score):
        # Load and preprocess image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0

        labels = {
            'disease': tf.cast(disease_id, tf.int32),
            'nutrient': tf.cast(nutrient_id, tf.int32),
            'health': tf.cast(health_score, tf.float32)
        }
        return image, labels

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation
    if augment:
        augmenter = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])

        def augment_fn(image, labels):
            image = augmenter(image, training=True)
            return image, labels

        dataset = dataset.shuffle(1000)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

class MultiTaskLoss(keras.losses.Loss):
    """Multi-task loss with proper class balancing"""
    
    def __init__(self, disease_weight=1.0, nutrient_weight=1.0, health_weight=0.3):
        super().__init__()
        self.disease_weight = disease_weight
        self.nutrient_weight = nutrient_weight
        self.health_weight = health_weight
        
        # Sparse categorical crossentropy for class indices
        self.disease_loss = keras.losses.SparseCategoricalCrossentropy()
        self.nutrient_loss = keras.losses.SparseCategoricalCrossentropy()
        self.health_loss = keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        """Calculate combined loss"""
        
        # Extract labels
        disease_true = y_true['disease']
        nutrient_true = y_true['nutrient']
        health_true = y_true['health']
        
        # Calculate individual losses
        disease_loss = self.disease_loss(disease_true, y_pred['disease'])
        nutrient_loss = self.nutrient_loss(nutrient_true, y_pred['nutrient'])
        health_loss = self.health_loss(
            tf.expand_dims(health_true, -1), 
            y_pred['health']
        )
        
        # Weighted combination
        total_loss = (
            self.disease_weight * disease_loss +
            self.nutrient_weight * nutrient_loss +
            self.health_weight * health_loss
        )
        
        return total_loss

def train():
    """Main training function - FIXED VERSION"""
    
    print("="*70)
    print("JIVA PLANTS: FIXED TRAINING PIPELINE")
    print("="*70)
    
    # Load class mappings
    print("\nLoading class mappings...")
    class_name_to_id, nutrient_name_to_id, unified_classes = load_class_mappings()
    
    # Load splits
    print("Loading data splits...")
    with open(PROCESSED_DIR / "splits.json") as f:
        splits = json.load(f)
    
    train_data = splits["train"]
    val_data = splits["val"]
    
    print(f"  Train: {len(train_data)} images")
    print(f"  Val:   {len(val_data)} images")
    
    # Create datasets
    print("\nCreating data pipelines...")
    train_ds = create_dataset(
        train_data,
        BATCH_SIZE,
        PROCESSED_DIR,  # Base dir - source_dataset field determines subdirectory
        augment=True,
        class_name_to_id=class_name_to_id,
        nutrient_name_to_id=nutrient_name_to_id,
    )

    val_ds = create_dataset(
        val_data,
        BATCH_SIZE,
        PROCESSED_DIR,
        augment=False,
        class_name_to_id=class_name_to_id,
        nutrient_name_to_id=nutrient_name_to_id,
    )
    
    print("  ✓ Data pipelines ready")
    
    # Create model
    print("\nCreating model...")
    num_disease_classes = len(class_name_to_id)
    num_nutrient_classes = len(nutrient_name_to_id)
    print(f"  Disease classes: {num_disease_classes}, Nutrient classes: {num_nutrient_classes}")

    model = create_model(
        num_disease_classes=num_disease_classes,
        num_nutrient_classes=num_nutrient_classes,
        img_size=IMG_SIZE
    )
    
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    print(f"  ✓ Model created")
    
    # Compile
    print("Compiling model...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=INITIAL_LR,
        weight_decay=1e-5
    )

    # Use per-output losses (required for multi-output models in Keras)
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
    
    print("  ✓ Model compiled")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_DIR / "model_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
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
        )
    ]
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Initial LR: {INITIAL_LR}")
    print(f"  Loss weights: Disease={DISEASE_LOSS_WEIGHT}, Nutrient={NUTRIENT_LOSS_WEIGHT}, Health={HEALTH_LOSS_WEIGHT}")
    
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    
    # Save final model
    model.save(str(CHECKPOINT_DIR / "model_final.h5"))
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Best model: {CHECKPOINT_DIR}/model_best.h5")
    print(f"Final model: {CHECKPOINT_DIR}/model_final.h5")
    
    return model, history

if __name__ == "__main__":
    model, history = train()