#!/usr/bin/env python3
"""
Convert all models to TFLite for iOS/Android deployment
Includes quantization for smaller file size
"""

import tensorflow as tf
from pathlib import Path
import json
import os

print("="*70)
print("CONVERTING MODELS TO TFLITE")
print("="*70)

# Ensure models directory exists
os.makedirs("./models", exist_ok=True)

# Track converted models for metadata
converted_models = {}

# ============= DISEASE MODEL =============
print("\n[1] Checking Disease Model...")

disease_model_path = Path("./checkpoints_disease_only/model_best.h5")
if disease_model_path.exists():
    print("  Loading disease model...")
    disease_model = tf.keras.models.load_model(str(disease_model_path))

    # Convert to TFLite (full precision)
    converter = tf.lite.TFLiteConverter.from_keras_model(disease_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True
    tflite_disease_full = converter.convert()

    with open("./models/disease_detector.tflite", "wb") as f:
        f.write(tflite_disease_full)

    print(f"  ✓ Full precision: disease_detector.tflite ({len(tflite_disease_full)/1024/1024:.1f} MB)")

    # Convert to TFLite (int8 quantized - smaller)
    converter = tf.lite.TFLiteConverter.from_keras_model(disease_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_disease_int8 = converter.convert()

    with open("./models/disease_detector_int8.tflite", "wb") as f:
        f.write(tflite_disease_int8)

    print(f"  ✓ Quantized (int8): disease_detector_int8.tflite ({len(tflite_disease_int8)/1024/1024:.1f} MB)")

    converted_models['disease_detector'] = {
        'full': 'disease_detector.tflite',
        'quantized': 'disease_detector_int8.tflite',
        'size_full_mb': len(tflite_disease_full) / 1024 / 1024,
        'size_quantized_mb': len(tflite_disease_int8) / 1024 / 1024,
        'classes': 21,
        'input_size': 224
    }
else:
    print("  ⚠ Disease model not found at checkpoints_disease_only/model_best.h5")
    print("    Skipping disease model conversion...")
    tflite_disease_full = b''
    tflite_disease_int8 = b''

# ============= NUTRIENT MODEL =============
print("\n[2] Converting Nutrient Model...")

# Check both possible paths (.keras and .h5)
nutrient_model_path = Path("./checkpoints_nutrient/model_best.keras")
if not nutrient_model_path.exists():
    nutrient_model_path = Path("./checkpoints_nutrient_enhanced/model_best.h5")

if nutrient_model_path.exists():
    print(f"  Loading nutrient model from {nutrient_model_path}...")
    # Load without compiling to avoid needing custom loss function
    nutrient_model = tf.keras.models.load_model(str(nutrient_model_path), compile=False)

    # Full precision
    converter = tf.lite.TFLiteConverter.from_keras_model(nutrient_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_nutrient_full = converter.convert()

    with open("./models/nutrient_detector.tflite", "wb") as f:
        f.write(tflite_nutrient_full)

    print(f"  ✓ Full precision: nutrient_detector.tflite ({len(tflite_nutrient_full)/1024/1024:.1f} MB)")

    # Quantized
    converter = tf.lite.TFLiteConverter.from_keras_model(nutrient_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_nutrient_int8 = converter.convert()

    with open("./models/nutrient_detector_int8.tflite", "wb") as f:
        f.write(tflite_nutrient_int8)

    print(f"  ✓ Quantized (int8): nutrient_detector_int8.tflite ({len(tflite_nutrient_int8)/1024/1024:.1f} MB)")

    converted_models['nutrient_detector'] = {
        'full': 'nutrient_detector.tflite',
        'quantized': 'nutrient_detector_int8.tflite',
        'size_full_mb': len(tflite_nutrient_full) / 1024 / 1024,
        'size_quantized_mb': len(tflite_nutrient_int8) / 1024 / 1024,
        'classes': 9,
        'input_size': 224
    }
else:
    print("  ⚠ Nutrient model not found")
    print("    Skipping nutrient model conversion...")
    tflite_nutrient_full = b''
    tflite_nutrient_int8 = b''

# ============= PLANT IDENTIFICATION MODEL =============
print("\n[3] Checking Plant Identification Model...")

plant_model_path = Path("./checkpoints_plant_identification/model_best.h5")
if plant_model_path.exists():
    print("  Loading plant identification model...")
    plant_model = tf.keras.models.load_model(str(plant_model_path))

    # Full precision
    converter = tf.lite.TFLiteConverter.from_keras_model(plant_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_plant_full = converter.convert()

    with open("./models/plant_identification.tflite", "wb") as f:
        f.write(tflite_plant_full)

    print(f"  ✓ Full precision: plant_identification.tflite ({len(tflite_plant_full)/1024/1024:.1f} MB)")

    # Quantized
    converter = tf.lite.TFLiteConverter.from_keras_model(plant_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_plant_int8 = converter.convert()

    with open("./models/plant_identification_int8.tflite", "wb") as f:
        f.write(tflite_plant_int8)

    print(f"  ✓ Quantized (int8): plant_identification_int8.tflite ({len(tflite_plant_int8)/1024/1024:.1f} MB)")

    converted_models['plant_identification'] = {
        'full': 'plant_identification.tflite',
        'quantized': 'plant_identification_int8.tflite',
        'size_full_mb': len(tflite_plant_full) / 1024 / 1024,
        'size_quantized_mb': len(tflite_plant_int8) / 1024 / 1024,
        'input_size': 224
    }
else:
    print("  ⚠ Plant ID model not found at checkpoints_plant_identification/model_best.h5")
    print("    Note: Plant ID uses HuggingFace model (juppy44/plant-identification-2m-vit-b)")
    print("    This model runs server-side and doesn't need TFLite conversion.")
    tflite_plant_full = b''
    tflite_plant_int8 = b''

# ============= SAVE METADATA =============
print("\n[4] Saving metadata...")

total_full = sum(m.get('size_full_mb', 0) for m in converted_models.values())
total_quantized = sum(m.get('size_quantized_mb', 0) for m in converted_models.values())

metadata = {
    'models': converted_models,
    'total_size_mb': total_full,
    'total_size_quantized_mb': total_quantized
}

with open("./models/tflite_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ============= SUMMARY =============
print("\n" + "="*70)
print("CONVERSION COMPLETE!")
print("="*70)

if converted_models:
    print("\nTFLite Models Ready for Mobile:")
    for name, info in converted_models.items():
        print(f"  {name}: {info['size_quantized_mb']:.1f} MB (quantized)")
    print(f"  ────────────────────────────")
    print(f"  TOTAL: {total_quantized:.1f} MB")
    print("\nReady for iOS/Android deployment!")
else:
    print("\nNo models were converted. Please ensure model checkpoint files exist.")

print("\nOutput files saved to ./models/")