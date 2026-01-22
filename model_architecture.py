#!/usr/bin/env python3
"""
Multi-task learning model for plant disease and nutrient deficiency detection
Architecture: EfficientNetV2-B2 backbone + ViT blocks + multi-task heads
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class EfficientNetV2ViTHybrid(keras.Model):
    """
    Hybrid model combining EfficientNetV2 backbone with Vision Transformer blocks
    and multi-task heads for disease, nutrient, and health detection
    """
    
    def __init__(
        self,
        num_disease_classes=21,
        num_nutrient_classes=9,
        backbone_name="efficientnetv2-b2",
        img_size=224,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_disease_classes = num_disease_classes
        self.num_nutrient_classes = num_nutrient_classes
        self.img_size = img_size
        
        # ==================== Stage 1: Feature Extraction ====================
        
        # Load pretrained EfficientNetV2 backbone
        self.backbone = keras.applications.EfficientNetV2B2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3)
        )
        self.backbone.trainable = False  # Freeze initially
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Feature expansion layer (to 512 dimensions)
        self.feature_expansion = keras.Sequential([
            layers.Dense(512, activation='relu', name='feature_expansion'),
            layers.BatchNormalization(name='feature_bn'),
            layers.Dropout(0.3, name='feature_dropout'),
        ], name='feature_module')
        
        # ==================== Stage 2: Vision Transformer Blocks ===========
        
        # Multi-head self-attention for global feature modeling
        self.mha_1 = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1,
            name='mha_1'
        )
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6, name='norm_1')
        
        self.mha_2 = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1,
            name='mha_2'
        )
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6, name='norm_2')
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(1024, activation='relu', name='ffn_dense1'),
            layers.Dropout(0.2, name='ffn_dropout'),
            layers.Dense(512, name='ffn_dense2'),
        ], name='ffn')
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6, name='norm_3')
        
        # ==================== Stage 3: Task-Specific Heads =================
        
        # Disease classification head
        self.disease_head = keras.Sequential([
            layers.Dense(256, activation='relu', name='disease_dense1'),
            layers.BatchNormalization(name='disease_bn1'),
            layers.Dropout(0.4, name='disease_dropout1'),
            layers.Dense(128, activation='relu', name='disease_dense2'),
            layers.BatchNormalization(name='disease_bn2'),
            layers.Dropout(0.3, name='disease_dropout2'),
            layers.Dense(num_disease_classes, activation='softmax', name='disease_output')
        ], name='disease_head')
        
        # Nutrient deficiency head
        self.nutrient_head = keras.Sequential([
            layers.Dense(256, activation='relu', name='nutrient_dense1'),
            layers.BatchNormalization(name='nutrient_bn1'),
            layers.Dropout(0.4, name='nutrient_dropout1'),
            layers.Dense(128, activation='relu', name='nutrient_dense2'),
            layers.BatchNormalization(name='nutrient_bn2'),
            layers.Dropout(0.3, name='nutrient_dropout2'),
            layers.Dense(num_nutrient_classes, activation='softmax', name='nutrient_output')
        ], name='nutrient_head')
        
        # Health score head (0-100)
        self.health_head = keras.Sequential([
            layers.Dense(256, activation='relu', name='health_dense1'),
            layers.BatchNormalization(name='health_bn1'),
            layers.Dropout(0.3, name='health_dropout1'),
            layers.Dense(128, activation='relu', name='health_dense2'),
            layers.Dense(1, activation='sigmoid', name='health_output')
        ], name='health_head')
    
    def call(self, inputs, training=None):
        """Forward pass"""
        
        # Stage 1: Feature extraction
        x = self.backbone(inputs, training=training)
        x = self.global_pool(x)
        features = self.feature_expansion(x, training=training)
        
        # Reshape for self-attention (add sequence dimension)
        x = tf.expand_dims(features, axis=1)  # Shape: (batch, 1, 512)
        
        # Stage 2: Vision Transformer blocks
        # First multi-head attention block
        attn_out_1 = self.mha_1(x, x, training=training)
        x = self.norm_1(x + attn_out_1)
        
        # Second multi-head attention block
        attn_out_2 = self.mha_2(x, x, training=training)
        x = self.norm_2(x + attn_out_2)
        
        # Feed-forward network
        ffn_out = self.ffn(x, training=training)
        x = self.norm_3(x + ffn_out)
        
        # Remove sequence dimension
        x = tf.squeeze(x, axis=1)  # Shape: (batch, 512)
        
        # Stage 3: Task-specific predictions
        disease_pred = self.disease_head(x, training=training)
        nutrient_pred = self.nutrient_head(x, training=training)
        health_pred = self.health_head(x, training=training)
        
        return {
            'disease': disease_pred,
            'nutrient': nutrient_pred,
            'health': health_pred
        }

def create_model(
    num_disease_classes=21,
    num_nutrient_classes=9,
    img_size=224,
):
    """Factory function to create model"""
    
    model = EfficientNetV2ViTHybrid(
        num_disease_classes=num_disease_classes,
        num_nutrient_classes=num_nutrient_classes,
        img_size=img_size,
        name='jiva_plants_multitask_v2'
    )
    
    return model

def get_model_summary(model, img_size=224):
    """Print model summary"""
    model.build(input_shape=(None, img_size, img_size, 3))
    model.summary()