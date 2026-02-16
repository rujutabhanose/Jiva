#!/usr/bin/env python3
"""
FIXED VALIDATION SCRIPT
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

PROCESSED_DIR = Path("./data/processed")
CHECKPOINT_DIR = Path("./checkpoints_disease_only")
IMG_SIZE = 224

print("="*70)
print("VALIDATING DISEASE MODEL")
print("="*70)

# ============= LOAD MODEL =============
print("\n[1] Loading model...")

model_path = CHECKPOINT_DIR / "model_best.h5"
if not model_path.exists():
    model_path = CHECKPOINT_DIR / "model_final.h5"

if not model_path.exists():
    print("❌ No model found!")
    exit(1)

model = tf.keras.models.load_model(str(model_path))
print(f"  ✓ Model loaded: {model_path.name}")

# ============= LOAD DATA =============
print("\n[2] Loading validation data...")

with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

val_data = splits['val']

# Load images
val_images = []
val_labels = []

for i, img_info in enumerate(val_data):
    if i % 500 == 0:
        print(f"  Loading {i}/{len(val_data)}")
    
    img_path = PROCESSED_DIR.parent / "raw" / img_info["path"]
    
    try:
        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        
        val_images.append(img)
        val_labels.append(img_info['class_id'])
    except:
        pass

val_images = np.array(val_images)
val_labels = np.array(val_labels)

print(f"  ✓ Loaded {len(val_images)} validation images")

# ============= LOAD CLASS NAMES =============
print("\n[3] Loading class mapping...")

with open(PROCESSED_DIR / "unified_class_index.json") as f:
    class_index = json.load(f)

# Convert to {id: name}
id_to_name = {int(k): v for k, v in class_index.items()}

print(f"  ✓ Loaded {len(id_to_name)} classes")

# ============= RUN PREDICTIONS =============
print("\n[4] Running predictions...")

predictions = model.predict(val_images, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)
predicted_probs = np.max(tf.nn.softmax(predictions), axis=1)

# ============= OVERALL ACCURACY =============
overall_acc = np.mean(predicted_classes == val_labels)

print(f"\n✅ OVERALL VALIDATION ACCURACY: {overall_acc:.4f} ({overall_acc*100:.1f}%)")

# ============= PER-CLASS ACCURACY =============
print("\n[5] Per-class accuracy:")

per_class_acc = {}
for class_id in sorted(np.unique(val_labels)):
    mask = val_labels == class_id
    class_acc = np.mean(predicted_classes[mask] == val_labels[mask])
    class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
    per_class_acc[class_id] = class_acc
    
    print(f"  {class_id:2d} | {class_name:40s} | {class_acc:.4f} ({class_acc*100:.1f}%)")

# ============= SUMMARY =============
print("\n" + "="*70)
min_acc = min(per_class_acc.values())
max_acc = max(per_class_acc.values())
avg_acc = np.mean(list(per_class_acc.values()))

print(f"Overall accuracy: {overall_acc:.1%}")
print(f"Average per-class: {avg_acc:.1%}")
print(f"Min per-class: {min_acc:.1%}")
print(f"Max per-class: {max_acc:.1%}")

if overall_acc > 0.90:
    print("\n✅ EXCELLENT! Ready for production.")
elif overall_acc > 0.80:
    print("\n✅ GOOD! Can be used with caution.")
elif overall_acc > 0.70:
    print("\n⚠️  NEEDS IMPROVEMENT")
else:
    print("\n❌ POOR. Check data pipeline.")

print("="*70)