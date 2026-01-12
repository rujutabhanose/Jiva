# train_models.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

class EfficientNetLiteHousePlantModel:
    """
    Multi-task EfficientNet-Lite-0 for:
    - Disease classification (21 classes)
    - Nutrient deficiency classification (9 classes)
    - Health score regression (0-100)
    """
    
    def __init__(self, input_shape=(224, 224, 3), disease_classes=21, nutrient_classes=9):
        self.input_shape = input_shape
        self.disease_classes = disease_classes
        self.nutrient_classes = nutrient_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build multi-task EfficientNet-Lite-0"""
        
        # Load pretrained EfficientNet-Lite-0
        # Available via TensorFlow Hub: https://tfhub.dev/google/lite-model/efficientnet/lite0/classification/2
        base_model = keras.applications.EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        
        # Freeze base layers (transfer learning)
        base_model.trainable = False
        
        # Input
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # Preprocessing (ImageNet normalization)
        x = keras.applications.efficientnet.preprocess_input(x)
        
        # Base model
        x = base_model(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Task 1: Disease Classification Head
        disease_branch = layers.Dense(128, activation='relu', name='disease_dense')(x)
        disease_branch = layers.Dropout(0.2)(disease_branch)
        disease_output = layers.Dense(
            self.disease_classes, 
            activation='softmax', 
            name='disease_output'
        )(disease_branch)
        
        # Task 2: Nutrient Deficiency Head
        nutrient_branch = layers.Dense(128, activation='relu', name='nutrient_dense')(x)
        nutrient_branch = layers.Dropout(0.2)(nutrient_branch)
        nutrient_output = layers.Dense(
            self.nutrient_classes, 
            activation='softmax', 
            name='nutrient_output'
        )(nutrient_branch)
        
        # Task 3: Health Score (Regression)
        health_branch = layers.Dense(64, activation='relu', name='health_dense')(x)
        health_output = layers.Dense(1, activation='sigmoid', name='health_score')(health_branch)
        health_output = layers.Lambda(lambda x: x * 100)(health_output)  # Scale to 0-100
        
        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs=[disease_output, nutrient_output, health_output],
            name='HousePlantDiagnoser'
        )
        
        print("✓ Model built successfully")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile with multi-task losses"""
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'disease_output': 'categorical_crossentropy',
                'nutrient_output': 'categorical_crossentropy',
                'health_score': 'mse'
            },
            loss_weights={
                'disease_output': 1.0,
                'nutrient_output': 0.8,
                'health_score': 0.2
            },
            metrics={
                'disease_output': 'accuracy',
                'nutrient_output': 'accuracy',
                'health_score': 'mae'
            }
        )
        
        print("✓ Model compiled")
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create train/val data generators with augmentation"""
        
        # Training augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation (minimal augmentation)
        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, train_dir, val_dir, epochs=30, batch_size=32):
        """Train the model"""
        
        train_gen, val_gen = self.prepare_data_generators(train_dir, val_dir, batch_size)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_disease_output_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("🚀 Starting training...")
        
        # Note: Need to create labeled datasets for multi-task learning
        # For now, simplified version - adapt based on your data structure
        
        # If using directory structure, may need custom data loading
        # self.history = self.model.fit(...)
        
        print("⚠️  Note: Implement custom data loader for multi-task labels")
        print("    Each image needs 3 labels: [disease_class, nutrient_class, health_score]")
    
    def save_model(self, filepath="house_plant_model.h5"):
        """Save trained model"""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def export_to_tflite(self, output_path="house_plant_model.tflite", quantize=True):
        """Export to TensorFlow Lite format"""
        
        print(f"📦 Exporting to TFLite {'with quantization' if quantize else ''}...")
        
        # Convert
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Dynamic range quantization (best for mobile)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        file_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ TFLite model exported: {file_size_mb:.2f} MB")
        
        return output_path


# Training Runner
if __name__ == "__main__":
    
    # Build model
    model_builder = EfficientNetLiteHousePlantModel(
        input_shape=(224, 224, 3),
        disease_classes=21,
        nutrient_classes=9
    )
    
    model_builder.build_model()
    model_builder.compile_model(learning_rate=0.001)
    
    # Print summary
    model_builder.model.summary()
    
    # Train (with your dataset paths)
    # model_builder.train(
    #     train_dir="./data/train",
    #     val_dir="./data/val",
    #     epochs=30,
    #     batch_size=32
    # )
    
    # Export to TFLite
    # model_builder.export_to_tflite(output_path="models/disease_detector.tflite")
    
    print("✅ Model pipeline ready!")