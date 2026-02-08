#!/usr/bin/env python3
"""Verify disease data is good before training"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import json
from pathlib import Path
from collections import Counter
import tensorflow as tf

PROCESSED_DIR = Path("./data/processed")

# Load splits
with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

train_data = splits['train']
val_data = splits['val']

print("="*70)
print("DATA VERIFICATION")
print("="*70)

# Check disease class distribution
print("\n[1] Disease Class Distribution (Train)")
disease_counts = Counter([img['class'] for img in train_data])

print(f"Total unique diseases: {len(disease_counts)}")
print(f"Total train samples: {len(train_data)}")

# Sort by count
for disease, count in sorted(disease_counts.items(), key=lambda x: -x[1])[:15]:
    pct = 100 * count / len(train_data)
    print(f"  {disease:40s}: {count:5d} ({pct:5.1f}%)")

min_samples = min(disease_counts.values())
max_samples = max(disease_counts.values())
imbalance = max_samples / min_samples

print(f"\nImbalance ratio: {imbalance:.1f}x (min={min_samples}, max={max_samples})")

if imbalance > 10:
    print("⚠️  WARNING: Class imbalance detected - will use class weights")
else:
    print("✓ Class distribution is reasonably balanced")

# Check class IDs
print("\n[2] Class ID Mapping")
class_ids = set([img['class_id'] for img in train_data])
print(f"Unique class IDs: {sorted(class_ids)}")
print(f"Min ID: {min(class_ids)}, Max ID: {max(class_ids)}")

if max(class_ids) == 18 and min(class_ids) == 0 and len(class_ids) == 19:
    print("✓ Class IDs are correct (0-18) for 19 classes")
else:
    print(f"❌ ERROR: Class IDs unexpected. Got IDs {sorted(class_ids)}")

# Check image paths
print("\n[3] Image Path Check (sampling 10 random)")
import random
sample_indices = random.sample(range(len(train_data)), min(10, len(train_data)))

missing = 0
for i in sample_indices:
    img_info = train_data[i]
    img_path = PROCESSED_DIR / img_info["path"]
    
    if img_path.exists():
        size_mb = img_path.stat().st_size / (1024*1024)
        print(f"  ✓ {img_path.name}: {size_mb:.1f} MB")
    else:
        print(f"  ❌ MISSING: {img_path}")
        missing += 1

if missing == 0:
    print(f"\n✓ All sampled paths exist")
else:
    print(f"\n❌ {missing}/10 paths missing!")

# Test image loading
print("\n[4] Image Loading Test")
try:
    sample_img = train_data[0]
    img_path = PROCESSED_DIR / sample_img["path"]
    
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Min: {tf.reduce_min(image).numpy():.1f}")
    print(f"  Max: {tf.reduce_max(image).numpy():.1f}")
    print(f"  ✓ Image loaded successfully")
    
except Exception as e:
    print(f"  ❌ Error: {e}")

# Validation set check
print("\n[5] Validation Set Stats")
val_disease_counts = Counter([img['class'] for img in val_data])
print(f"  Val samples: {len(val_data)}")
print(f"  Val unique classes: {len(val_disease_counts)}")
print(f"  Train/Val ratio: {len(train_data)/len(val_data):.1f}:1")

print("\n" + "="*70)
print("READY TO TRAIN" if missing == 0 else "❌ FIX DATA ISSUES FIRST")
print("="*70)