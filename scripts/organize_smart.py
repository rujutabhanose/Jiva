#!/usr/bin/env python3
"""
Organize filtered PlantVillage into train/val/test/calibration splits
Output: data/organized/
  - train/      (70%)
  - val/        (15%)
  - test/       (15%)
  - calibration (40 per class for model calibration)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

DATA_DIR = Path("data")
FILTERED_DIR = DATA_DIR / "plantvillage_filtered"
ORGANIZED_DIR = DATA_DIR / "organized"

def organize_data():
    """Split data into train/val/test/calibration."""
    print("🔧 Organizing dataset into train/val/test/calibration...\n")
    
    # Create directories
    for split in ["train", "val", "test", "calibration"]:
        (ORGANIZED_DIR / split).mkdir(parents=True, exist_ok=True)
    
    class_stats = {}
    
    for disease_dir in sorted(FILTERED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue
        
        class_name = disease_dir.name
        print(f"  Processing: {class_name}")
        
        # Create class subdirs
        for split in ["train", "val", "test", "calibration"]:
            (ORGANIZED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        images = sorted(
            list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.JPG"))
        )
        images = list(set(images))  # Remove duplicates
        
        if len(images) == 0:
            print(f"    ⚠️  No images found")
            continue
        
        # Split: reserve 40 for calibration, 70/15/15 for train/val/test
        calib_imgs = images[:min(40, len(images))]
        remaining = images[min(40, len(images)):]
        
        train_imgs, temp = train_test_split(
            remaining, test_size=0.30, random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp, test_size=0.50, random_state=42
        )
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(img, ORGANIZED_DIR / "train" / class_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, ORGANIZED_DIR / "val" / class_name / img.name)
        for img in test_imgs:
            shutil.copy2(img, ORGANIZED_DIR / "test" / class_name / img.name)
        for img in calib_imgs:
            shutil.copy2(img, ORGANIZED_DIR / "calibration" / class_name / img.name)
        
        class_stats[class_name] = {
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
            "calibration": len(calib_imgs)
        }
        
        print(f"    ✅ {len(train_imgs)}T {len(val_imgs)}V {len(test_imgs)}S {len(calib_imgs)}C")
    
    # Save stats
    with open(ORGANIZED_DIR / "stats.json", "w") as f:
        json.dump(class_stats, f, indent=2)
    
    print_stats(class_stats)
    return ORGANIZED_DIR, class_stats

def print_stats(stats):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    total_images = sum(s["total"] for s in stats.values())
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total_test = sum(s["test"] for s in stats.values())
    
    print(f"\nClasses: {len(stats)}")
    print(f"Total Images: {total_images:,}")
    print(f"  Training:   {total_train:,} ({100*total_train/total_images:.1f}%)")
    print(f"  Validation: {total_val:,} ({100*total_val/total_images:.1f}%)")
    print(f"  Test:       {total_test:,} ({100*total_test/total_images:.1f}%)")
    print(f"  Calibration: {40*len(stats)}")
    
    storage_gb = (total_images * 0.4) / 1024
    print(f"\nEstimated Storage: {storage_gb:.2f} GB")
    
    print("\nDiseases:")
    for cls_name in sorted(stats.keys()):
        print(f"  - {cls_name}")
    
    print("=" * 70 + "\n")

def main():
    os.chdir(Path(__file__).parent.parent)
    try:
        organized, stats = organize_data()
        print(f"✅ PHASE 0.2 COMPLETE!\n")
        print(f"Location: {organized}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        exit(1)

if __name__ == "__main__":
    main()