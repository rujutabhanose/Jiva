#!/usr/bin/env python3
"""
Convert all models to TFLite for iOS/Android deployment
Includes quantization for smaller file size
"""

import tensorflow as tf
from pathlib import Path
import json

print("="*70)
print("CONVERTING MODELS TO TFLITE")
print("="*70)

# ============= DISEASE MODEL =============
print("\n[1] Converting Disease Model...")

disease_model_path = "./checkpoints_disease_only/model_best.h5"
disease_model = tf.keras.models.load_model(disease_model_path)

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

# ============= NUTRIENT MODEL =============
print("\n[2] Converting Nutrient Model...")

nutrient_model_path = "./checkpoints_nutrient_enhanced/model_best.h5"
nutrient_model = tf.keras.models.load_model(nutrient_model_path)

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

# ============= PLANT IDENTIFICATION MODEL =============
print("\n[3] Converting Plant Identification Model...")

plant_model_path = "./checkpoints_plant_identification/model_best.h5"
plant_model = tf.keras.models.load_model(plant_model_path)

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

# ============= SAVE METADATA =============
print("\n[4] Saving metadata...")

metadata = {
    'models': {
        'disease_detector': {
            'full': 'disease_detector.tflite',
            'quantized': 'disease_detector_int8.tflite',
            'size_full_mb': len(tflite_disease_full) / 1024 / 1024,
            'size_quantized_mb': len(tflite_disease_int8) / 1024 / 1024,
            'classes': 21,
            'input_size': 224
        },
        'nutrient_detector': {
            'full': 'nutrient_detector.tflite',
            'quantized': 'nutrient_detector_int8.tflite',
            'size_full_mb': len(tflite_nutrient_full) / 1024 / 1024,
            'size_quantized_mb': len(tflite_nutrient_int8) / 1024 / 1024,
            'classes': 9,
            'input_size': 224
        },
        'plant_identification': {
            'full': 'plant_identification.tflite',
            'quantized': 'plant_identification_int8.tflite',
            'size_full_mb': len(tflite_plant_full) / 1024 / 1024,
            'size_quantized_mb': len(tflite_plant_int8) / 1024 / 1024,
            'input_size': 224
        }
    },
    'total_size_mb': (len(tflite_disease_full) + len(tflite_nutrient_full) + len(tflite_plant_full)) / 1024 / 1024,
    'total_size_quantized_mb': (len(tflite_disease_int8) + len(tflite_nutrient_int8) + len(tflite_plant_int8)) / 1024 / 1024
}

with open("./models/tflite_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ============= SUMMARY =============
print("\n" + "="*70)
print("CONVERSION COMPLETE!")
print("="*70)

print("\n📱 TFLite Models Ready for Mobile:")
print(f"  Disease:    {len(tflite_disease_int8)/1024/1024:.1f} MB (quantized)")
print(f"  Nutrient:   {len(tflite_nutrient_int8)/1024/1024:.1f} MB (quantized)")
print(f"  Plant ID:   {len(tflite_plant_int8)/1024/1024:.1f} MB (quantized)")
print(f"  ────────────────────────────")
print(f"  TOTAL:      {metadata['total_size_quantized_mb']:.1f} MB")

print("\n✅ Ready for iOS/Android deployment!")