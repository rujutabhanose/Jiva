"""
Model Trainer Service

Handles fine-tuning of models on user-confirmed training data.
Uses transfer learning with frozen base layers to prevent catastrophic forgetting.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json
import asyncio

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import httpx

from sqlalchemy.orm import Session

from app.models.training_data import TrainingData
from app.services.training_data_service import TrainingDataService
from app.services.model_manager import model_manager, MODELS_DIR, VERSIONED_MODELS_DIR
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model fine-tuning on user-confirmed samples.

    Training approach:
    1. Load existing model (transfer learning base)
    2. Freeze base layers to prevent forgetting
    3. Fine-tune top layers on new data
    4. Validate on holdout set
    5. Save and optionally activate new version
    """

    def __init__(self, db: Session):
        self.db = db
        self.training_service = TrainingDataService(db)
        self.img_size = 224

    async def fine_tune_coleaf(
        self,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        auto_activate: bool = False,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Fine-tune the CoLeaf nutrient deficiency model.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
            auto_activate: Whether to auto-activate if accuracy improves

        Returns:
            Tuple of (success, message, metrics_dict)
        """
        return await self._fine_tune_model(
            model_type="coleaf",
            base_model_path=MODELS_DIR / "coleaf_production_v2.keras",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            auto_activate=auto_activate,
        )

    async def fine_tune_disease(
        self,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        auto_activate: bool = False,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Fine-tune the disease detection model.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
            auto_activate: Whether to auto-activate if accuracy improves

        Returns:
            Tuple of (success, message, metrics_dict)
        """
        return await self._fine_tune_model(
            model_type="disease",
            base_model_path=MODELS_DIR / "disease_detector.h5",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            auto_activate=auto_activate,
        )

    async def _fine_tune_model(
        self,
        model_type: str,
        base_model_path: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        auto_activate: bool,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Generic fine-tuning logic for any model type.
        """
        try:
            # Check if we have enough samples
            if not self.training_service.can_retrain(model_type):
                unused_count = len(self.training_service.get_unused_samples(model_type))
                return (
                    False,
                    f"Insufficient training data: {unused_count}/{settings.MIN_TRAINING_SAMPLES} samples",
                    None,
                )

            logger.info(f"Starting fine-tuning for {model_type} model")

            # Prepare training data
            X_train, X_val, y_train, y_val, class_names, sample_ids = await self._prepare_data(
                model_type
            )

            if X_train is None or len(X_train) == 0:
                return False, "Failed to prepare training data", None

            logger.info(
                f"Prepared {len(X_train)} training and {len(X_val)} validation samples"
            )

            # Load base model
            if not base_model_path.exists():
                # Try to get from model manager
                base_model = model_manager.get_model(model_type)
                if base_model is None:
                    return False, f"Base model not found: {base_model_path}", None
            else:
                base_model = tf.keras.models.load_model(str(base_model_path))

            # Prepare model for fine-tuning
            model = self._prepare_for_finetuning(
                base_model, len(class_names), learning_rate
            )

            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest",
            )

            # Training callbacks
            early_stop = EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            )

            # Train
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1,
            )

            # Evaluate
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

            # Check if we should rollback
            if model_manager.should_rollback(val_accuracy, model_type):
                return (
                    False,
                    f"New model accuracy ({val_accuracy:.2%}) too low compared to current model",
                    {"validation_accuracy": val_accuracy},
                )

            # Save new version
            version = self._generate_version_string(model_type)
            save_path = VERSIONED_MODELS_DIR / f"{model_type}_{version}.keras"
            model.save(str(save_path))

            # Register version
            model_version = model_manager.register_new_version(
                db=self.db,
                model_type=model_type,
                version=version,
                file_path=str(save_path),
                accuracy=val_accuracy,
                validation_accuracy=val_accuracy,
                training_samples_count=len(X_train),
                epochs_trained=len(history.history["loss"]),
                notes=f"Fine-tuned on {len(X_train)} user-confirmed samples",
                activate=auto_activate,
            )

            if model_version:
                # Mark samples as used
                self.training_service.mark_samples_as_used(sample_ids, model_version.id)

            metrics = {
                "model_type": model_type,
                "version": version,
                "validation_accuracy": val_accuracy,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "epochs_trained": len(history.history["loss"]),
                "class_distribution": {
                    name: int(np.sum(y_train.argmax(axis=1) == i))
                    for i, name in enumerate(class_names)
                },
                "activated": auto_activate,
            }

            return True, f"Successfully fine-tuned {model_type} model to {version}", metrics

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return False, f"Fine-tuning failed: {str(e)}", None

    async def _prepare_data(
        self, model_type: str
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        List[str],
        List[int],
    ]:
        """
        Prepare training data from confirmed samples.

        Returns:
            Tuple of (X_train, X_val, y_train, y_val, class_names, sample_ids)
        """
        try:
            samples = self.training_service.get_unused_samples(model_type)

            if not samples:
                return None, None, None, None, [], []

            # Collect images and labels
            images = []
            labels = []
            sample_ids = []
            label_to_idx = {}

            for sample in samples:
                # Download/load image
                img_array = await self._load_image(sample.image_url)
                if img_array is not None:
                    images.append(img_array)

                    # Map label to index
                    if sample.confirmed_label not in label_to_idx:
                        label_to_idx[sample.confirmed_label] = len(label_to_idx)
                    labels.append(label_to_idx[sample.confirmed_label])
                    sample_ids.append(sample.id)

            if len(images) < 10:
                logger.warning(f"Only {len(images)} valid images found")
                return None, None, None, None, [], []

            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)

            # One-hot encode labels
            num_classes = len(label_to_idx)
            y_onehot = tf.keras.utils.to_categorical(y, num_classes)

            # Split into train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_onehot, test_size=settings.TRAINING_VALIDATION_SPLIT, random_state=42
            )

            class_names = list(label_to_idx.keys())

            return X_train, X_val, y_train, y_val, class_names, sample_ids

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None, None, None, None, [], []

    async def _load_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an image from URL or base64.

        Args:
            image_url: URL or base64-encoded image

        Returns:
            Preprocessed image array or None
        """
        try:
            if image_url.startswith("http"):
                # Download from URL
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    if response.status_code != 200:
                        return None
                    image_data = response.content
            elif image_url.startswith("data:"):
                # Base64 encoded
                import base64
                _, encoded = image_url.split(",", 1)
                image_data = base64.b64decode(encoded)
            else:
                # Local file path
                if os.path.exists(image_url):
                    with open(image_url, "rb") as f:
                        image_data = f.read()
                else:
                    return None

            # Load and preprocess
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name

            img = Image.open(tmp_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size))
            arr = np.array(img).astype("float32") / 255.0

            os.unlink(tmp_path)
            return arr

        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return None

    def _prepare_for_finetuning(
        self, model: tf.keras.Model, num_classes: int, learning_rate: float
    ) -> tf.keras.Model:
        """
        Prepare a model for fine-tuning by freezing base layers.

        Args:
            model: Base model to fine-tune
            num_classes: Number of output classes
            learning_rate: Learning rate

        Returns:
            Model prepared for fine-tuning
        """
        # Freeze all layers except the last few
        for layer in model.layers[:-5]:
            layer.trainable = False

        # Unfreeze last few layers for fine-tuning
        for layer in model.layers[-5:]:
            layer.trainable = True

        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _generate_version_string(self, model_type: str) -> str:
        """
        Generate a version string for a new model.

        Args:
            model_type: Type of model

        Returns:
            Version string (e.g., "v1.2.0_20240115")
        """
        # Get current version count
        existing_versions = model_manager.get_all_versions(self.db, model_type)
        major_version = len(existing_versions) + 1

        timestamp = datetime.utcnow().strftime("%Y%m%d")
        return f"v{major_version}.0.0_{timestamp}"


async def run_training_job(
    db: Session,
    model_type: str,
    epochs: int = 10,
    auto_activate: bool = False,
) -> Dict[str, Any]:
    """
    Run a training job for a specific model type.

    Args:
        db: Database session
        model_type: Type of model to train
        epochs: Number of training epochs
        auto_activate: Whether to auto-activate if successful

    Returns:
        Dictionary with training results
    """
    trainer = ModelTrainer(db)

    if model_type == "coleaf":
        success, message, metrics = await trainer.fine_tune_coleaf(
            epochs=epochs, auto_activate=auto_activate
        )
    elif model_type == "disease":
        success, message, metrics = await trainer.fine_tune_disease(
            epochs=epochs, auto_activate=auto_activate
        )
    else:
        return {
            "success": False,
            "message": f"Unknown model type: {model_type}",
            "metrics": None,
        }

    return {
        "success": success,
        "message": message,
        "metrics": metrics,
    }
