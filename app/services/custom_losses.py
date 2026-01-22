# app/services/custom_losses.py
"""
Custom loss functions for Keras models.

These must be registered before loading any models that use them.
"""

import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance in nutrient deficiency detection."""

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), 9)  # 9 classes

        # Focal loss formula
        ce_loss = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * ce_loss

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma
        })
        return config
