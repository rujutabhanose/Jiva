"""
Centralized config for all phases
"""

from dataclasses import dataclass
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class DataConfig:
    """Data configuration."""
    ROOT = PROJECT_ROOT / "data" / "organized"
    
    TRAIN = ROOT / "train"
    VAL = ROOT / "val"
    TEST = ROOT / "test"
    CALIBRATION = ROOT / "calibration"
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    AUGMENTATION = True
    
    # Load class mapping
    @staticmethod
    def get_class_mapping():
        """Load class name mapping."""
        mapping_file = PROJECT_ROOT / "disease_class_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                return json.load(f)
        return None

@dataclass
class ModelConfig:
    """Model training parameters."""
    # TFLite (on-device)
    TFLITE_EPOCHS = 10
    TFLITE_LR = 1e-4
    TFLITE_BATCH = 32
    TFLITE_PATIENCE = 2
    TFLITE_BACKBONE = "MobileNetV2"
    
    # Ensemble (cloud)
    ENSEMBLE_EPOCHS = 10
    ENSEMBLE_LR = 1e-4
    ENSEMBLE_BATCH = 16
    ENSEMBLE_PATIENCE = 2
    ENSEMBLE_BACKBONES = ["VGG16", "ResNet50", "InceptionV3"]

@dataclass
class PathConfig:
    """Model & output paths."""
    ROOT = PROJECT_ROOT / "models"
    
    TFLITE = ROOT / "tflite"
    ENSEMBLE = ROOT / "ensemble"
    CHECKPOINTS = ROOT / "checkpoints"
    
    CALIBRATORS = ROOT / "calibrators.pkl"
    WEIGHTS = ROOT / "model_weights.json"
    KB = ROOT / "knowledge_base.json"
    
    def __post_init__(self):
        """Create directories."""
        for path in [self.TFLITE, self.ENSEMBLE, self.CHECKPOINTS]:
            path.mkdir(parents=True, exist_ok=True)

# Global instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
PATH_CONFIG = PathConfig()