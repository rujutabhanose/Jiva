#!/usr/bin/env python3
"""
FIXED DATA PROCESSING - HANDLES 15 CLASSES FROM PLANTVILLAGE
This processes PlantVillage data with proper class mapping
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import random

# ============= CONFIG =============
# Use absolute path to the PlantVillage data
SCRIPT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "PlantVillage"
PROCESSED_DIR = DATA_DIR / "processed"
IMG_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("DATA PROCESSING - 15 CLASS MAPPING")
print("="*70)

# ============= DEFINE 15 CLASSES FROM YOUR DATASET =============
# These match the actual folders in PlantVillage
DISEASE_CLASSES = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy",
}

# Reverse mapping
CLASS_NAME_TO_ID = {v: k for k, v in DISEASE_CLASSES.items()}

NUM_CLASSES = len(DISEASE_CLASSES)
print(f"\n✓ {NUM_CLASSES} classes defined:")
for cid, cname in sorted(DISEASE_CLASSES.items()):
    print(f"  {cid:2d} → {cname}")

# ============= SCAN RAW DATA =============
print(f"\n\nScanning {RAW_DIR}...")

all_images = []
class_counts = {i: 0 for i in range(NUM_CLASSES)}

# Look for folders matching class names
for class_folder in sorted(RAW_DIR.glob("*")):
    if not class_folder.is_dir():
        continue
    
    folder_name = class_folder.name
    
    # Find matching class ID
    class_id = None
    for cid, cname in DISEASE_CLASSES.items():
        if cname.lower() == folder_name.lower() or \
           cname.replace("___", "_").lower() == folder_name.lower():
            class_id = cid
            break
    
    if class_id is None:
        print(f"  ⚠️  Skipping folder '{folder_name}' (no class match)")
        continue
    
    # Load all images in this folder
    image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + \
                  list(class_folder.glob("*.png")) + list(class_folder.glob("*.JPG"))
    
    for img_path in image_files:
        try:
            # Verify image is valid
            img = Image.open(img_path)
            img.verify()
            
            all_images.append({
                'path': f"{folder_name}/{img_path.name}",
                'class_id': class_id,
                'class_name': DISEASE_CLASSES[class_id],
            })
            class_counts[class_id] += 1
        except Exception as e:
            print(f"    ✗ Invalid: {img_path.name}")

print(f"\n✓ Scanned {len(all_images)} valid images")
print("\nClass distribution:")
for cid in sorted(class_counts.keys()):
    count = class_counts[cid]
    pct = 100 * count / len(all_images)
    bar = "█" * max(1, int(pct / 2))
    print(f"  Class {cid:2d}: {count:5d} ({pct:5.1f}%) {bar}")

# ============= SPLIT DATA =============
print("\n\nSplitting data...")

train_data, val_data = train_test_split(
    all_images,
    test_size=0.18,
    random_state=SEED,
    stratify=[img['class_id'] for img in all_images]
)

print(f"  Train: {len(train_data)} images")
print(f"  Val:   {len(val_data)} images")

# ============= SAVE SPLITS =============
print("\nSaving splits...")

splits = {
    'train': train_data,
    'val': val_data,
}

with open(PROCESSED_DIR / "splits.json", 'w') as f:
    json.dump(splits, f, indent=2)

print("  ✓ splits.json saved")

# ============= SAVE CLASS INDEX =============
print("Saving class index...")

class_index = DISEASE_CLASSES.copy()

with open(PROCESSED_DIR / "unified_class_index.json", 'w') as f:
    json.dump(class_index, f, indent=2)

print("  ✓ unified_class_index.json saved")

# ============= VERIFY =============
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

train_class_ids = [img['class_id'] for img in train_data]
val_class_ids = [img['class_id'] for img in val_data]

print(f"\nUnique classes in train: {sorted(set(train_class_ids))}")
print(f"Unique classes in val:   {sorted(set(val_class_ids))}")
print(f"All {NUM_CLASSES} classes present: {sorted(set(train_class_ids + val_class_ids)) == list(range(NUM_CLASSES))}")

print("\n✅ DATA PROCESSING COMPLETE")
print("="*70)