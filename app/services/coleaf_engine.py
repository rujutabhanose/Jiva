# app/services/coleaf_engine.py
"""
CoLeaf nutrient deficiency classifier integration.

Model loading priority:
  1. TFLite quantized (nutrient_detector_int8.tflite) - fastest, smallest
  2. TFLite full (nutrient_detector.tflite) - fallback
  3. Keras model (coleaf_production_v2.keras) - fallback (requires full tensorflow)

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

from app.services.model_manager import model_manager
from app.services.custom_losses import FocalLoss

logger = logging.getLogger(__name__)

# TFLite interpreter: prefer tflite_runtime, fall back to full tensorflow
try:
    import tflite_runtime.interpreter as tflite
    _HAS_TFLITE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        _HAS_TFLITE = True
    except ImportError:
        _HAS_TFLITE = False

# Full tensorflow for Keras model loading (training environments only)
try:
    import tensorflow as tf
    from tensorflow import keras
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.info("TensorFlow not available — Keras model fallback disabled (TFLite inference only)")

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


# TFLite model paths
TFLITE_QUANTIZED = MODELS_DIR / "nutrient_detector_int8.tflite"
TFLITE_FULL = MODELS_DIR / "nutrient_detector.tflite"

# Model state
_tflite_interpreter = None
_keras_model = None
_model_format = None  # "tflite" or "keras"


def _load_tflite_interpreter():
    """Load TFLite interpreter with fallback between quantized and full models."""
    global _tflite_interpreter, _model_format

    if _tflite_interpreter is not None:
        return _tflite_interpreter

    if not _HAS_TFLITE:
        logger.warning("No TFLite runtime available for CoLeaf")
        return None

    for model_path in [TFLITE_QUANTIZED, TFLITE_FULL]:
        if model_path.exists():
            try:
                interpreter = tflite.Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                _tflite_interpreter = interpreter
                _model_format = "tflite"
                logger.info(f"CoLeaf TFLite model loaded: {model_path.name}")
                return interpreter
            except Exception as e:
                logger.warning(f"Failed to load TFLite {model_path.name}: {e}")
                continue

    return None


def _load_keras_model():
    """Load Keras model as fallback (requires full tensorflow)."""
    global _keras_model, _model_format

    if _keras_model is not None:
        return _keras_model

    if not _HAS_TF:
        logger.info("Keras model fallback skipped — tensorflow not installed")
        return None

    try:
        _keras_model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={'FocalLoss': FocalLoss}
        )
        _model_format = "keras"
        logger.info(f"CoLeaf Keras model loaded: {MODEL_FILE.name}")
        return _keras_model
    except Exception as e:
        logger.error(f"Failed to load CoLeaf Keras model: {e}")
        return None


# Initialize: Try TFLite first, then Keras
_load_tflite_interpreter() or _load_keras_model()
if _model_format:
    logger.info(f"CoLeaf engine initialized with {_model_format} model: {len(CLASS_NAMES)} classes")
else:
    logger.error("CoLeaf engine: No model loaded!")


def _get_model_and_format() -> tuple:
    """
    Get the active CoLeaf model (TFLite or Keras) with format info.

    Returns:
        Tuple of (model_or_interpreter, format_type)
        format_type is "tflite" or "keras"
    """
    # Try model manager first (for versioned models)
    if model_manager.is_tflite("coleaf"):
        interpreter = model_manager.get_tflite_interpreter("coleaf")
        if interpreter:
            return interpreter, "tflite"

    managed_model = model_manager.get_model("coleaf")
    if managed_model is not None:
        return managed_model, "keras"

    # Fallback to locally loaded models
    if _tflite_interpreter is not None:
        return _tflite_interpreter, "tflite"

    if _keras_model is not None:
        return _keras_model, "keras"

    return None, None


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


def _run_tflite_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run inference using TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])[0]


def run_coleaf(image_path: str) -> Dict[str, Any]:
    """
    Run CoLeaf nutrient classifier and return a diagnosis dict.

    Uses TFLite model if available, falls back to Keras.
    This is designed to be merged with MobileNet results inside
    diagnosis_engine.diagnose_image().

    Returns empty result if model failed to load.
    """
    # Get model (TFLite or Keras)
    model, model_format = _get_model_and_format()

    # Return empty result if model failed to load
    if model is None:
        logger.warning("CoLeaf model not loaded - skipping nutrient deficiency detection")
        return {"diagnoses": [], "confidence": 0.0, "model_version_id": None}

    x = _preprocess_image(image_path)

    # Run inference based on model format
    if model_format == "tflite":
        preds = _run_tflite_inference(model, x)
        logger.debug("CoLeaf inference via TFLite")
    else:
        preds = model.predict(x, verbose=0)[0]
        logger.debug("CoLeaf inference via Keras")

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
                "model_format": model_format,  # Track which format was used
            }
        ],
        "confidence": conf,
        "model_version_id": model_version_id,
        "model_format": model_format,
    }
