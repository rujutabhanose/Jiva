"""
TFLite runtime compatibility layer.

Tries to import the TFLite Interpreter from (in order):
  1. ai_edge_litert (new Google LiteRT package)
  2. tflite_runtime (legacy standalone package)
  3. tensorflow.lite (full tensorflow)

Usage:
    from app.services.tflite_compat import TFLiteInterpreter, HAS_TFLITE
    if HAS_TFLITE:
        interpreter = TFLiteInterpreter(model_path="model.tflite")
"""

import logging

logger = logging.getLogger(__name__)

TFLiteInterpreter = None
HAS_TFLITE = False

try:
    from ai_edge_litert.interpreter import Interpreter
    TFLiteInterpreter = Interpreter
    HAS_TFLITE = True
    logger.info("Using ai-edge-litert for TFLite inference")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        TFLiteInterpreter = Interpreter
        HAS_TFLITE = True
        logger.info("Using tflite-runtime for TFLite inference")
    except ImportError:
        try:
            import tensorflow as tf
            TFLiteInterpreter = tf.lite.Interpreter
            HAS_TFLITE = True
            logger.info("Using tensorflow.lite for TFLite inference")
        except ImportError:
            logger.warning("No TFLite runtime available — model inference disabled")
