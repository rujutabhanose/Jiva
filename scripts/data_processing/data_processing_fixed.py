#!/usr/bin/env python3
"""
FIXED DATA PROCESSING - Handles nested dataset folder structure
Processes: plant-village/segmented, plantify-dr, niphad-grape
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import random

# ============= CONFIG =============
DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMG_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DATA PROCESSING - NESTED FOLDER STRUCTURE")
print("=" * 70)

# ============= DATASET CONFIGURATIONS =============
# Each entry: (dataset_name, raw_path, class_name_mapping)
# class_name_mapping: dict mapping folder names to standardized class names (or None to use folder name as-is)

DATASETS = [
    ("plant_village", RAW_DIR / "plant-village" / "segmented", None),
    ("plantify_dr", RAW_DIR / "plantify-dr", None),
    ("niphad_grape", RAW_DIR / "niphad-grape", None),
]

# ============= BUILD GLOBAL CLASS MAPPING =============
print("\nScanning datasets to build class mapping...")

class_name_to_id = {}
next_id = 0
all_images = []

for dataset_name, raw_path, name_mapping in DATASETS:
    if not raw_path.exists():
        print(f"  Warning: {raw_path} not found, skipping")
        continue

    print(f"\nProcessing: {dataset_name} ({raw_path})")

    # Create output directory for this dataset
    dataset_out_dir = PROCESSED_DIR / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all class folders
    class_folders = sorted([d for d in raw_path.iterdir() if d.is_dir()])

    for class_folder in tqdm(class_folders, desc=f"  {dataset_name}"):
        folder_name = class_folder.name

        # Get standardized class name
        if name_mapping and folder_name in name_mapping:
            class_name = name_mapping[folder_name]
        else:
            class_name = folder_name

        # Assign class ID if new
        if class_name not in class_name_to_id:
            class_name_to_id[class_name] = next_id
            next_id += 1

        class_id = class_name_to_id[class_name]

        # Create output class directory
        class_out_dir = dataset_out_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
            image_files.extend(class_folder.glob(ext))

        for img_path in image_files:
            try:
                # Load and resize image
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

                # Generate output filename
                img_index = len(all_images)
                out_filename = f"{img_index:06d}.jpg"
                out_path = class_out_dir / out_filename

                # Save processed image
                img.save(out_path, quality=95)

                # Record image info
                all_images.append({
                    "id": img_index,
                    "class": class_name,
                    "class_id": class_id,
                    "path": f"{dataset_name}/{class_name}/{out_filename}",
                    "source_dataset": dataset_name,
                })

            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")

print(f"\n\nTotal images processed: {len(all_images)}")
print(f"Total classes found: {len(class_name_to_id)}")

# ============= CLASS DISTRIBUTION =============
print("\nClass distribution:")
from collections import Counter
class_counts = Counter(img['class_id'] for img in all_images)

for class_name, class_id in sorted(class_name_to_id.items(), key=lambda x: x[1]):
    count = class_counts.get(class_id, 0)
    pct = 100 * count / len(all_images) if all_images else 0
    bar = "#" * max(1, int(pct / 2))
    print(f"  {class_id:2d} {class_name:45s} {count:5d} ({pct:5.1f}%) {bar}")

# ============= SPLIT DATA =============
print("\n\nSplitting data (82% train, 18% val)...")

if len(all_images) == 0:
    print("ERROR: No images found! Check your data/raw directory structure.")
    exit(1)

train_data, val_data = train_test_split(
    all_images,
    test_size=0.18,
    random_state=SEED,
    stratify=[img['class_id'] for img in all_images]
)

print(f"  Train: {len(train_data)} images")
print(f"  Val:   {len(val_data)} images")

# Add split field
for img in train_data:
    img['split'] = 'train'
for img in val_data:
    img['split'] = 'val'

# ============= SAVE FILES =============
print("\nSaving output files...")

# Save splits
splits = {'train': train_data, 'val': val_data}
with open(PROCESSED_DIR / "splits.json", 'w') as f:
    json.dump(splits, f)
print("  splits.json")

# Save unified class index (id -> name)
unified_class_index = {str(v): k for k, v in class_name_to_id.items()}
with open(PROCESSED_DIR / "unified_class_index.json", 'w') as f:
    json.dump(unified_class_index, f, indent=2)
print("  unified_class_index.json")

# Save class name to id mapping
with open(PROCESSED_DIR / "class_name_to_id.json", 'w') as f:
    json.dump(class_name_to_id, f, indent=2)
print("  class_name_to_id.json")

# Save all images index
with open(PROCESSED_DIR / "unified_image_index.json", 'w') as f:
    json.dump(all_images, f)
print("  unified_image_index.json")

# ============= VERIFICATION =============
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

train_classes = set(img['class_id'] for img in train_data)
val_classes = set(img['class_id'] for img in val_data)

print(f"\nClasses in train: {sorted(train_classes)}")
print(f"Classes in val:   {sorted(val_classes)}")
print(f"Total unique classes: {len(class_name_to_id)}")

# Verify a few paths exist
print("\nVerifying sample paths...")
for img in random.sample(all_images, min(5, len(all_images))):
    full_path = PROCESSED_DIR / img['path']
    status = "OK" if full_path.exists() else "MISSING"
    print(f"  [{status}] {img['path']}")

print("\n" + "=" * 70)
print("DATA PROCESSING COMPLETE")
print("=" * 70)
