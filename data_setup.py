#!/usr/bin/env python3
"""
Data setup script - Downloads and organizes all datasets
Run once to prepare all training data
"""

import os
import json
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

# Configuration
DATA_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("JIVA PLANTS: DATA SETUP")
print("=" * 70)

# Step 1: PlantVillage Dataset
print("\n[1/5] PlantVillage Dataset")
print("      Manual download required (too large for automated download)")
print("      URL: https://github.com/spMohanty/PlantVillage-Dataset")
print("      OR:  kaggle datasets download -d arjuntejaswi/plant-village")
print("      Extract to: ./data/raw/plant-village/")
print("")
print("      Expected structure:")
print("      ./data/raw/plant-village/segmented/")
print("      ├── Apple___Apple_scab/")
print("      ├── Apple___Black_rot/")
print("      └── ... (38 classes)")
input("      Press ENTER after download and extraction...")

# Step 2: PlantifyDr Dataset
print("\n[2/5] PlantifyDr Dataset (Kaggle)")
print("      Command: kaggle datasets download -d pratikgarai/plantify-plant-disease-detection-dataset")
print("      Extract to: ./data/raw/plantify-dr/")
input("      Press ENTER after download and extraction...")

# Step 3: Niphad Grape Dataset
print("\n[3/5] Niphad Grape Dataset (Mendeley)")
print("      URL: https://data.mendeley.com/datasets/f3mfg7krb7")
print("      Extract to: ./data/raw/niphad-grape/")
input("      Press ENTER after download and extraction...")

# Step 4: Verify Downloads
print("\n[4/5] Verifying Downloads...")
datasets = {
    "PlantVillage": DATA_DIR / "plant-village" / "segmented",
    "PlantifyDr": DATA_DIR / "plantify-dr",
    "Niphad Grape": DATA_DIR / "niphad-grape"
}

for name, path in datasets.items():
    if path.exists():
        num_classes = len([d for d in path.iterdir() if d.is_dir()])
        num_images = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.JPG"))) + \
                     len(list(path.rglob("*.png"))) + len(list(path.rglob("*.PNG")))
        print(f"   ✓ {name}: {num_classes} classes, {num_images} images")
    else:
        print(f"   ✗ {name}: NOT FOUND at {path}")

print("\n[5/5] Creating index files...")
# Will be done in data_processing.py
print("   Done! Run data_processing.py next")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Run: python data_processing.py")
print("2. Then: python prepare_splits.py")
print("=" * 70)