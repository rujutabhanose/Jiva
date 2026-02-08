#!/usr/bin/env python3
"""
Final diagnostic: Find EXACTLY what's wrong if accuracy stays stuck
Run this if validation accuracy doesn't improve after epoch 5
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter

PROCESSED_DIR = Path("./data/processed")

print("="*70)
print("DEEP DIAGNOSTIC: Finding The Problem")
print("="*70)

# 1. Load splits
print("\n[1] Loading splits...")
with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

train_data = splits['train']
val_data = splits['val']

print(f"  Train samples: {len(train_data)}")
print(f"  Val samples: {len(val_data)}")

# 2. Check class_id distribution (THIS IS CRITICAL)
print("\n[2] CLASS ID DISTRIBUTION (This must be 0-20, not all zeros!)")

train_class_ids = [img['class_id'] for img in train_data]
class_id_dist = Counter(train_class_ids)

print(f"\n  Number of unique class IDs: {len(class_id_dist)}")
print(f"  Class IDs found: {sorted(class_id_dist.keys())}")

if len(class_id_dist) == 1:
    print(f"\n  ❌ CRITICAL ERROR: ALL class IDs are {list(class_id_dist.keys())[0]}!")
    print(f"     This means data_processing.py failed!")
    print(f"     FIX: rm -rf ./data/processed/* && python data_processing_fixed.py")
    exit(1)

if max(class_id_dist.keys()) <= 9:
    print(f"\n  ❌ ERROR: Class IDs only go up to {max(class_id_dist.keys())}")
    print(f"     Expected: 0-20 for 21 disease classes")
    print(f"     FIX: Rebuild data processing")
    exit(1)

print(f"\n  ✓ Class IDs look correct (0-{max(class_id_dist.keys())})")

# 3. Class distribution by ID
print("\n[3] DISTRIBUTION OF SAMPLES BY CLASS ID")

for class_id in sorted(class_id_dist.keys()):
    count = class_id_dist[class_id]
    pct = 100 * count / len(train_data)
    bar = "█" * int(pct / 2)
    print(f"  Class {class_id:2d}: {count:5d} ({pct:5.1f}%) {bar}")

min_samples = min(class_id_dist.values())
max_samples = max(class_id_dist.values())
imbalance_ratio = max_samples / min_samples

print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}x")
if imbalance_ratio > 100:
    print(f"  ⚠️  SEVERE imbalance - class weighting REQUIRED")

# 4. Map class_id back to class_name
print("\n[4] MAPPING CLASS ID → CLASS NAME")

class_id_to_name = {}
for img in train_data[:100]:  # Sample first 100
    if img['class_id'] not in class_id_to_name:
        class_id_to_name[img['class_id']] = img['class']

for class_id in sorted(class_id_to_name.keys())[:10]:
    class_name = class_id_to_name[class_id]
    count = class_id_dist[class_id]
    print(f"  {class_id:2d} → {class_name:40s} ({count} samples)")

# 5. Verify unified_class_index.json
print("\n[5] CHECKING unified_class_index.json")

with open(PROCESSED_DIR / "unified_class_index.json") as f:
    unified_index = json.load(f)

print(f"  Total classes in index: {len(unified_index)}")

# Should have 21 disease classes
expected_diseases = {
    'apple_scab', 'apple_black_rot', 'apple_cedar_rust',
    'blueberry_healthy',
    'cherry_healthy', 'cherry_powdery_mildew',
    'corn_cercospora_leaf_spot', 'corn_common_rust', 'corn_northern_leaf_blight', 'corn_healthy',
    'grape_black_rot', 'grape_esca', 'grape_leaf_blight', 'grape_healthy',
    'peach_bacterial_spot_canker', 'peach_healthy',
    'pepper_bacterial_spot', 'pepper_healthy',
    'potato_early_blight', 'potato_late_blight', 'potato_healthy',
}

actual_diseases = set(unified_index.values())

missing = expected_diseases - actual_diseases
extra = actual_diseases - expected_diseases

if missing:
    print(f"\n  ❌ MISSING classes: {missing}")

if len(actual_diseases) != 21:
    print(f"\n  ❌ ERROR: Expected 21 classes, found {len(actual_diseases)}")
    print(f"  Classes found:")
    for idx, name in sorted(unified_index.items()):
        print(f"    {idx:2s} → {name}")
else:
    print(f"  ✓ All 21 disease classes present")

# 6. Test image loading
print("\n[6] TESTING IMAGE LOADING & PREPROCESSING")

sample_indices = np.random.choice(len(train_data), 5, replace=False)
failed = 0

for idx in sample_indices:
    img_info = train_data[idx]
    img_path = PROCESSED_DIR / img_info["path"]

    try:
        image = tf.io.read_file(str(img_path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        
        min_val = float(tf.reduce_min(image).numpy())
        max_val = float(tf.reduce_max(image).numpy())
        mean_val = float(tf.reduce_mean(image).numpy())
        
        # Check image is reasonable
        if max_val < 0.01:
            print(f"  ⚠️  Image {idx} is all black (max={max_val:.4f})")
        
        if min_val < 0 or max_val > 1.0:
            print(f"  ⚠️  Image {idx} has out-of-range values: [{min_val:.4f}, {max_val:.4f}]")
        
    except Exception as e:
        print(f"  ❌ Failed to load image {idx}: {e}")
        failed += 1

if failed == 0:
    print(f"  ✓ All sampled images loaded successfully")

# 7. Check batch creation
print("\n[7] TESTING BATCH CREATION")

try:
    def load_image(img_info):
        img_path = PROCESSED_DIR / img_info["path"]
        image = tf.io.read_file(str(img_path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(img_info['class_id'], tf.int32)
        return image, label
    
    dataset = tf.data.Dataset.from_generator(
        lambda: [load_image(img) for img in train_data[:100]],
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    dataset = dataset.batch(32)
    
    batch_images, batch_labels = next(iter(dataset))
    
    print(f"  Batch image shape: {batch_images.shape}")
    print(f"  Batch label shape: {batch_labels.shape}")
    print(f"  Label values: min={tf.reduce_min(batch_labels)}, max={tf.reduce_max(batch_labels)}")
    print(f"  ✓ Batch creation successful")
    
except Exception as e:
    print(f"  ❌ Batch creation failed: {e}")

# 8. Final verdict
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

all_good = True

if len(class_id_dist) == 1:
    print("❌ FAIL: Only 1 unique class ID (all data is same class)")
    print("   FIX: rm -rf ./data/processed/* && python data_processing_fixed.py")
    all_good = False

elif len(class_id_dist) < 20:
    print(f"❌ FAIL: Only {len(class_id_dist)} unique class IDs (need 21)")
    print("   FIX: Check data_processing_fixed.py")
    all_good = False

elif len(unified_index) != 21:
    print(f"❌ FAIL: unified_class_index.json has {len(unified_index)} classes (need 21)")
    print("   FIX: Rebuild data")
    all_good = False

elif failed > 0:
    print(f"❌ FAIL: {failed}/5 images failed to load")
    print("   FIX: Check image files")
    all_good = False

if all_good:
    print("✅ SUCCESS: Data looks perfect!")
    print("\nYour accuracy should improve quickly:")
    print("  Epoch 1-5:   5% → 35%")
    print("  Epoch 5-20:  35% → 70%")
    print("  Epoch 20-50: 70% → 90%+")
    print("\nIf it doesn't, the problem is in train_disease_only.py")

print("="*70)