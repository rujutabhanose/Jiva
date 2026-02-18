# app/services/custom_losses.py
"""
Custom loss functions for Keras models.

These must be registered before loading any models that use them.
When tensorflow is not available (tflite-runtime only), provides a
placeholder so imports don't break.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.info("TensorFlow not available — FocalLoss only usable during training (GitHub Actions)")

if _HAS_TF:
    @tf.keras.utils.register_keras_serializable()
    class FocalLoss(keras.losses.Loss):
        """Focal Loss for handling class imbalance in nutrient deficiency detection."""

        def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.gamma = gamma

        def call(self, y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            if len(y_true.shape) == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), 9)
            ce_loss = -y_true * tf.math.log(y_pred)
            focal_weight = tf.pow(1 - y_pred, self.gamma)
            focal_loss = self.alpha * focal_weight * ce_loss
            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

        def get_config(self):
            config = super().get_config()
            config.update({"alpha": self.alpha, "gamma": self.gamma})
            return config
else:
    class FocalLoss:
        """Placeholder — only used when loading Keras models (requires full tensorflow)."""
        def __init__(self, **kwargs):
            pass
