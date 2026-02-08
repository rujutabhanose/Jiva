# app/services/coleaf_engine.py
"""
CoLeaf nutrient deficiency classifier integration.

Uses TensorFlow/Keras model with versioning support:
  - Active version loaded via ModelManager
  - Fallback to baseline: backend/models/coleaf_production_v2.keras

Config:
  backend/models/class_indices.json

Output is compatible with diagnosis_engine._merge_results():
  {
    "diagnoses": [
      {
        "source": "coleaf",
        "raw_class": "N",
        "label": "nitrogen_deficiency",
        "confidence": 0.93,
        "category": "nutrient_deficiency",
        "model_version_id": 1,
      }
    ],
    "confidence": 0.93,
  }
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

# Use standard TensorFlow/Keras (model saved with Functional API in Keras 3.x)
import tensorflow as tf
from tensorflow import keras

from app.services.model_manager import model_manager
from app.services.custom_losses import FocalLoss

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]   # .../backend
MODELS_DIR = BASE_DIR / "models"

CLASS_INDICES_FILE = MODELS_DIR / "class_indices.json"
MODEL_FILE = MODELS_DIR / "coleaf_production_v2.keras"
MODEL_CONFIG_FILE = MODELS_DIR / "model_config.json"


# Load class indices (new format from class_indices.json)
with CLASS_INDICES_FILE.open("r", encoding="utf-8") as f:
    CLASS_INDICES: Dict[str, int] = json.load(f)

# Create reverse mapping: index -> nutrient name
# Example: {0: "B", 1: "Ca", 2: "Fe", ...}
CLASS_NAMES = [""] * len(CLASS_INDICES)
for nutrient, idx in CLASS_INDICES.items():
    CLASS_NAMES[idx] = nutrient

IMG_SIZE = 224  # fixed for this model

# Map short nutrient codes to your disease_mapping keys
NUTRIENT_TO_LABEL: Dict[str, str] = {
    "N": "nitrogen_deficiency",
    "P": "phosphorus_deficiency",
    "K": "potassium_deficiency",
    "Mg": "magnesium_deficiency",
    "Ca": "calcium_deficiency",
    "Fe": "iron_deficiency",
    # Not explicitly modelled in DISEASE_MAPPINGS yet → use general fallback
    "Mn": "general_nutrient_deficiency",
    "B": "general_nutrient_deficiency",
    "Zn": "general_nutrient_deficiency",  # Zinc deficiency
}


# Fallback: Load Keras model once at import (used if model_manager not initialized)
coleaf_model = None
try:
    # Use standard TensorFlow Keras loading for the new production model
    # Pass custom_objects for FocalLoss used during training
    coleaf_model = tf.keras.models.load_model(
        MODEL_FILE,
        custom_objects={'FocalLoss': FocalLoss}
    )
    logger.info(f"CoLeaf baseline model loaded: {len(CLASS_NAMES)} classes")
except Exception as e:
    logger.error(f"Failed to load CoLeaf baseline model: {e}")


def _get_model() -> Optional[Any]:
    """
    Get the active CoLeaf model from model manager or fallback.

    Returns:
        Loaded Keras model or None
    """
    # Try model manager first (for versioned models)
    managed_model = model_manager.get_model("coleaf")
    if managed_model is not None:
        return managed_model

    # Fallback to baseline model
    return coleaf_model


def get_model_version_id() -> Optional[int]:
    """
    Get the current model version ID.

    Returns:
        Model version ID or None if using baseline
    """
    return model_manager.get_model_version_id("coleaf")


def _preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def run_coleaf(image_path: str) -> Dict[str, Any]:
    """
    Run CoLeaf nutrient classifier and return a diagnosis dict.

    This is designed to be merged with MobileNet results inside
    diagnosis_engine.diagnose_image().

    Returns empty result if model failed to load.
    """
    # Get model (from manager or fallback)
    model = _get_model()

    # Return empty result if model failed to load
    if model is None:
        logger.warning("CoLeaf model not loaded - skipping nutrient deficiency detection")
        return {"diagnoses": [], "confidence": 0.0, "model_version_id": None}

    x = _preprocess_image(image_path)
    preds = model.predict(x, verbose=0)[0]  # (num_classes,)

    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    cls = CLASS_NAMES[idx]  # e.g. "N", "Fe", "Healthy"

    if cls == "Healthy":
        label = "general_nutrient_deficiency"  # or "general_plant_stress"
        category = "nutrient_deficiency"
    else:
        label = NUTRIENT_TO_LABEL.get(cls, "general_nutrient_deficiency")
        category = "nutrient_deficiency"

    # Get model version for tracking
    model_version_id = get_model_version_id()

    return {
        "diagnoses": [
            {
                "source": "coleaf",
                "raw_class": cls,      # original CoLeaf class code
                "label": label,        # must exist in DISEASE_MAPPINGS
                "confidence": conf,
                "category": category,
                "model_version_id": model_version_id,
            }
        ],
        "confidence": conf,
        "model_version_id": model_version_id,
    }