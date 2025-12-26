# app/services/coleaf_engine.py
"""
CoLeaf nutrient deficiency classifier integration.

Uses TensorFlow/Keras model:
  backend/models/coleaf_classifier_final.h5

Config:
  backend/models/class_mapping.json

Output is compatible with diagnosis_engine._merge_results():
  {
    "diagnoses": [
      {
        "source": "coleaf",
        "raw_class": "N",
        "label": "nitrogen_deficiency",
        "confidence": 0.93,
        "category": "nutrient_deficiency",
      }
    ],
    "confidence": 0.93,
  }
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

# Use standard TensorFlow/Keras (model saved with Functional API in Keras 3.x)
import tensorflow as tf


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


# Load Keras model once at import
try:
    # Use standard TensorFlow Keras loading for the new production model
    coleaf_model = tf.keras.models.load_model(MODEL_FILE)
    import logging
    logging.info(f"✓ CoLeaf model loaded successfully: {len(CLASS_NAMES)} classes")
except Exception as e:
    import logging
    logging.error(f"Failed to load CoLeaf model: {e}")
    coleaf_model = None


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
    # Return empty result if model failed to load
    if coleaf_model is None:
        import logging
        logging.warning("CoLeaf model not loaded - skipping nutrient deficiency detection")
        return {"diagnoses": [], "confidence": 0.0}

    x = _preprocess_image(image_path)
    preds = coleaf_model.predict(x, verbose=0)[0]  # (num_classes,)

    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    cls = CLASS_NAMES[idx]  # e.g. "N", "Fe", "Healthy"

    if cls == "Healthy":
        label = "general_nutrient_deficiency"  # or "general_plant_stress"
        category = "nutrient_deficiency"
    else:
        label = NUTRIENT_TO_LABEL.get(cls, "general_nutrient_deficiency")
        category = "nutrient_deficiency"

    return {
        "diagnoses": [
            {
                "source": "coleaf",
                "raw_class": cls,      # original CoLeaf class code
                "label": label,        # must exist in DISEASE_MAPPINGS
                "confidence": conf,
                "category": category,
            }
        ],
        "confidence": conf,
    }