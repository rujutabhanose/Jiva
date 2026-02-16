"""
Model Manager Service

Handles model versioning, loading, hot-swapping, and rollback.
Enables continuous learning by managing multiple model versions.

Supports:
  - TFLite models (.tflite) - Primary for mobile compatibility
  - Keras models (.keras, .h5) - Fallback
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import tensorflow as tf
from sqlalchemy.orm import Session

from app.models.model_version import ModelVersion
from app.core.config import settings
from app.services.custom_losses import FocalLoss

logger = logging.getLogger(__name__)

# Base directory for models
BASE_DIR = Path(__file__).resolve().parents[2]  # .../backend
MODELS_DIR = BASE_DIR / "models"
VERSIONED_MODELS_DIR = MODELS_DIR / "versions"

# TFLite model paths
TFLITE_MODELS = {
    "disease": {
        "quantized": MODELS_DIR / "disease_detector_int8.tflite",
        "full": MODELS_DIR / "disease_detector.tflite",
    },
    "coleaf": {
        "quantized": MODELS_DIR / "nutrient_detector_int8.tflite",
        "full": MODELS_DIR / "nutrient_detector.tflite",
    },
}


class ModelManager:
    """
    Manages multiple versions of ML models with hot-swap capability.

    Features:
    - Load active model version at startup
    - Hot-swap models without server restart
    - Automatic rollback if accuracy drops
    - Thread-safe model access
    - TFLite support with Keras fallback
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}  # model_type -> loaded model (Keras)
        self._tflite_interpreters: Dict[str, Any] = {}  # model_type -> TFLite interpreter
        self._model_types: Dict[str, str] = {}  # model_type -> "tflite" or "keras"
        self._model_versions: Dict[str, ModelVersion] = {}  # model_type -> version info
        self._lock = threading.RLock()
        self._initialized = False

        # Ensure versioned models directory exists
        VERSIONED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def initialize(self, db: Session):
        """
        Initialize model manager by loading active versions from database.

        Args:
            db: Database session
        """
        with self._lock:
            if self._initialized:
                return

            logger.info("Initializing Model Manager...")

            for model_type in ["coleaf", "disease", "plant_id"]:
                self._load_active_model(db, model_type)

            self._initialized = True
            logger.info("Model Manager initialized successfully")

    def _load_active_model(self, db: Session, model_type: str) -> bool:
        """
        Load the active model version for a given model type.

        Args:
            db: Database session
            model_type: Type of model (coleaf, disease, plant_id)

        Returns:
            True if model loaded successfully
        """
        try:
            # Get active version from database
            active_version = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.is_active == True,
                )
                .first()
            )

            if active_version:
                # Load the versioned model
                model_path = Path(active_version.file_path)
                if model_path.exists():
                    model = self._load_model_file(model_path, model_type)
                    if model:
                        self._models[model_type] = model
                        self._model_versions[model_type] = active_version
                        logger.info(
                            f"Loaded {model_type} model version {active_version.version}"
                        )
                        return True
                else:
                    logger.warning(
                        f"Model file not found: {model_path}, using baseline"
                    )

            # Fall back to baseline model
            return self._load_baseline_model(model_type)

        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            return self._load_baseline_model(model_type)

    def _load_baseline_model(self, model_type: str) -> bool:
        """
        Load the baseline model with TFLite priority and Keras fallback.

        Priority:
        1. TFLite quantized (smallest, fastest)
        2. TFLite full precision
        3. Keras/H5 model

        Args:
            model_type: Type of model

        Returns:
            True if loaded successfully
        """
        # Try TFLite first (mobile-compatible)
        if model_type in TFLITE_MODELS:
            tflite_paths = TFLITE_MODELS[model_type]
            for variant in ["quantized", "full"]:
                path = tflite_paths.get(variant)
                if path and path.exists():
                    interpreter = self._load_tflite_model(path)
                    if interpreter:
                        self._tflite_interpreters[model_type] = interpreter
                        self._model_types[model_type] = "tflite"
                        logger.info(f"Loaded TFLite {model_type} model ({variant}): {path.name}")
                        return True

        # Fallback to Keras/H5
        baseline_paths = {
            "coleaf": MODELS_DIR / "coleaf_production_v2.keras",
            "disease": MODELS_DIR / "disease_detector.h5",
            "plant_id": None,  # Uses HuggingFace, no local file
        }

        path = baseline_paths.get(model_type)
        if path and path.exists():
            model = self._load_model_file(path, model_type)
            if model:
                self._models[model_type] = model
                self._model_types[model_type] = "keras"
                logger.info(f"Loaded Keras baseline {model_type} model: {path.name}")
                return True

        logger.warning(f"No baseline model available for {model_type}")
        return False

    def _load_tflite_model(self, path: Path) -> Optional[Any]:
        """
        Load a TFLite model.

        Args:
            path: Path to .tflite file

        Returns:
            TFLite interpreter or None
        """
        try:
            interpreter = tf.lite.Interpreter(model_path=str(path))
            interpreter.allocate_tensors()
            logger.info(f"TFLite model loaded: {path.name}")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to load TFLite model {path}: {e}")
            return None

    def _load_model_file(self, path: Path, model_type: str) -> Optional[Any]:
        """
        Load a model file based on its extension.

        Args:
            path: Path to model file
            model_type: Type of model

        Returns:
            Loaded model or None
        """
        try:
            if path.suffix == ".tflite":
                return self._load_tflite_model(path)
            elif path.suffix in [".keras", ".h5"]:
                # Pass custom_objects for models that use custom loss functions
                custom_objects = {'FocalLoss': FocalLoss}
                return tf.keras.models.load_model(str(path), custom_objects=custom_objects)
            else:
                logger.error(f"Unknown model format: {path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return None

    def get_model(self, model_type: str) -> Optional[Any]:
        """
        Get the currently active Keras model for a given type.

        Args:
            model_type: Type of model

        Returns:
            Loaded Keras model or None
        """
        with self._lock:
            return self._models.get(model_type)

    def get_tflite_interpreter(self, model_type: str) -> Optional[Any]:
        """
        Get the TFLite interpreter for a given model type.

        Args:
            model_type: Type of model

        Returns:
            TFLite interpreter or None
        """
        with self._lock:
            return self._tflite_interpreters.get(model_type)

    def get_model_type(self, model_type: str) -> Optional[str]:
        """
        Get the loaded model format type.

        Args:
            model_type: Type of model

        Returns:
            "tflite" or "keras" or None
        """
        with self._lock:
            return self._model_types.get(model_type)

    def is_tflite(self, model_type: str) -> bool:
        """Check if the loaded model is TFLite format."""
        return self.get_model_type(model_type) == "tflite"

    def get_model_version(self, model_type: str) -> Optional[ModelVersion]:
        """
        Get the version info for the currently active model.

        Args:
            model_type: Type of model

        Returns:
            ModelVersion record or None
        """
        with self._lock:
            return self._model_versions.get(model_type)

    def get_model_version_id(self, model_type: str) -> Optional[int]:
        """
        Get the ID of the currently active model version.

        Args:
            model_type: Type of model

        Returns:
            Model version ID or None
        """
        version = self.get_model_version(model_type)
        return version.id if version else None

    def hot_swap_model(
        self, db: Session, model_type: str, new_version_id: int
    ) -> Tuple[bool, str]:
        """
        Hot-swap to a different model version without restart.

        Args:
            db: Database session
            model_type: Type of model
            new_version_id: ID of the new version to activate

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            try:
                # Get the new version
                new_version = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.id == new_version_id)
                    .first()
                )

                if not new_version:
                    return False, f"Model version {new_version_id} not found"

                if new_version.model_type != model_type:
                    return False, f"Version {new_version_id} is not a {model_type} model"

                # Load the new model
                model_path = Path(new_version.file_path)
                if not model_path.exists():
                    return False, f"Model file not found: {model_path}"

                new_model = self._load_model_file(model_path, model_type)
                if not new_model:
                    return False, "Failed to load new model"

                # Deactivate current version
                db.query(ModelVersion).filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.is_active == True,
                ).update({ModelVersion.is_active: False})

                # Activate new version
                new_version.is_active = True
                db.commit()

                # Swap in memory
                old_model = self._models.get(model_type)
                self._models[model_type] = new_model
                self._model_versions[model_type] = new_version

                # Clean up old model (if not baseline)
                if old_model:
                    del old_model
                    tf.keras.backend.clear_session()

                logger.info(
                    f"Hot-swapped {model_type} model to version {new_version.version}"
                )
                return True, f"Successfully activated version {new_version.version}"

            except Exception as e:
                db.rollback()
                logger.error(f"Hot-swap failed: {e}")
                return False, f"Hot-swap failed: {str(e)}"

    def rollback_model(self, db: Session, model_type: str) -> Tuple[bool, str]:
        """
        Rollback to the previous model version.

        Args:
            db: Database session
            model_type: Type of model

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the two most recent versions
            versions = (
                db.query(ModelVersion)
                .filter(ModelVersion.model_type == model_type)
                .order_by(ModelVersion.created_at.desc())
                .limit(2)
                .all()
            )

            if len(versions) < 2:
                return False, "No previous version available for rollback"

            current = versions[0]
            previous = versions[1]

            if not current.is_active:
                # Find the active one
                active = (
                    db.query(ModelVersion)
                    .filter(
                        ModelVersion.model_type == model_type,
                        ModelVersion.is_active == True,
                    )
                    .first()
                )
                if active:
                    # Get the version before active
                    previous = (
                        db.query(ModelVersion)
                        .filter(
                            ModelVersion.model_type == model_type,
                            ModelVersion.id < active.id,
                        )
                        .order_by(ModelVersion.created_at.desc())
                        .first()
                    )
                    if not previous:
                        return False, "No previous version available"

            return self.hot_swap_model(db, model_type, previous.id)

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False, f"Rollback failed: {str(e)}"

    def register_new_version(
        self,
        db: Session,
        model_type: str,
        version: str,
        file_path: str,
        accuracy: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
        training_samples_count: int = 0,
        epochs_trained: Optional[int] = None,
        notes: Optional[str] = None,
        activate: bool = False,
    ) -> Optional[ModelVersion]:
        """
        Register a new model version in the database.

        Args:
            db: Database session
            model_type: Type of model
            version: Version string (e.g., "v1.1.0")
            file_path: Path to the model file
            accuracy: Overall accuracy
            validation_accuracy: Validation set accuracy
            training_samples_count: Number of samples used for training
            epochs_trained: Number of training epochs
            notes: Optional notes about this version
            activate: Whether to activate this version immediately

        Returns:
            Created ModelVersion record or None
        """
        try:
            model_version = ModelVersion(
                model_type=model_type,
                version=version,
                file_path=file_path,
                accuracy=accuracy,
                validation_accuracy=validation_accuracy,
                training_samples_count=training_samples_count,
                epochs_trained=epochs_trained,
                is_active=False,
                is_baseline=False,
                notes=notes,
            )

            db.add(model_version)
            db.commit()
            db.refresh(model_version)

            logger.info(f"Registered new {model_type} model version: {version}")

            if activate:
                success, msg = self.hot_swap_model(db, model_type, model_version.id)
                if not success:
                    logger.warning(f"Auto-activation failed: {msg}")

            return model_version

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to register model version: {e}")
            return None

    def should_rollback(
        self, new_accuracy: float, model_type: str
    ) -> bool:
        """
        Check if new model should be rolled back based on accuracy drop.

        Args:
            new_accuracy: Accuracy of the new model
            model_type: Type of model

        Returns:
            True if rollback is recommended
        """
        current_version = self.get_model_version(model_type)
        if not current_version or not current_version.accuracy:
            return False

        accuracy_drop = current_version.accuracy - new_accuracy
        if accuracy_drop > settings.MAX_ACCURACY_DROP:
            logger.warning(
                f"Accuracy drop of {accuracy_drop:.2%} exceeds threshold "
                f"({settings.MAX_ACCURACY_DROP:.2%}) for {model_type}"
            )
            return True

        return False

    def get_all_versions(self, db: Session, model_type: str) -> list:
        """
        Get all versions of a model type.

        Args:
            db: Database session
            model_type: Type of model

        Returns:
            List of ModelVersion records
        """
        return (
            db.query(ModelVersion)
            .filter(ModelVersion.model_type == model_type)
            .order_by(ModelVersion.created_at.desc())
            .all()
        )


# Global instance
model_manager = ModelManager()