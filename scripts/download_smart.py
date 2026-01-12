#!/usr/bin/env python3
"""
Smart PlantVillage downloader for 2GB constraint
Downloads only target crops: Tomato, Potato, Mango, Grape, Chili, Bean
Total: ~6,000 images, 22 disease classes
"""

import os
import subprocess
import json
from pathlib import Path
from PIL import Image
import shutil
import zipfile

# Configuration
DATA_DIR = Path("data")
BACKUP_DIR = Path("data_backup")
TARGET_CROPS = ["Tomato", "Potato", "Mango", "Grape", "Chili", "Bean"]
IMAGES_PER_CLASS = 250
TARGET_SIZE = 600  # Max 600px to save space

def setup_kaggle():
    """Verify Kaggle API is configured."""
    print("⚙️  Verifying Kaggle API...")
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_path.exists():
        print("❌ Kaggle API not configured!")
        print("\nFollow these steps:")
        print("1. Go to: https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        raise FileNotFoundError("Kaggle API not configured")
    
    print("✅ Kaggle API configured\n")

def download_plantvillage():
    """Download PlantVillage dataset from Kaggle."""
    print("📥 Downloading PlantVillage dataset...")
    print("   This may take 10-30 minutes depending on internet\n")
    
    try:
        # Download
        result = subprocess.run([
            "kaggle", "datasets", "download", "-d",
            "vipoooool/new-plant-diseases-dataset",
            "-p", str(DATA_DIR),
            "--unzip"
        ], capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        print("✅ Downloaded and extracted\n")
        
        # Find the extracted directory
        for item in DATA_DIR.iterdir():
            if "Plant Diseases" in item.name and item.is_dir():
                return item / item.name if (item / item.name).exists() else item
        
        return DATA_DIR / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
    
    except Exception as e:
        print(f"❌ Download failed: {e}\n")
        print("📌 MANUAL DOWNLOAD ALTERNATIVE:")
        print("   1. Go to: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset")
        print("   2. Click 'Download' button")
        print("   3. Extract to: data/")
        print("   4. Re-run this script\n")
        raise

def filter_by_crop(source_dir, target_crops, limit_per_class=250):
    """Filter only target crops and diseases to save space."""
    print(f"🔍 Filtering for {len(target_crops)} crops...")
    print(f"   Max {limit_per_class} images per disease class\n")
    
    filtered_dir = DATA_DIR / "plantvillage_filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    disease_count = 0
    
    for disease_dir in source_dir.iterdir():
        if not disease_dir.is_dir():
            continue
        
        # Parse: "Tomato___Early_blight" → crop="Tomato", disease="Early_blight"
        parts = disease_dir.name.split("___")
        if len(parts) != 2:
            continue
        
        crop = parts
        disease = parts
        
        # Keep only target crops
        if crop not in target_crops:
            continue
        
        # Create output directory
        out_dir = filtered_dir / disease_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        images = list(disease_dir.glob("*.JPG")) + list(disease_dir.glob("*.jpg"))
        images = images[:limit_per_class]
        
        for img_path in images:
            try:
                shutil.copy2(img_path, out_dir / img_path.name)
                total_count += 1
            except Exception as e:
                print(f"  ⚠️  Error copying {img_path.name}: {e}")
        
        size_mb = (len(images) * 0.4)
        print(f"  ✅ {disease_dir.name}: {len(images)} images ({size_mb:.0f} MB)")
        disease_count += 1
    
    total_gb = (total_count * 0.4) / 1024
    print(f"\n✅ Filtered {total_count} images in {disease_count} classes ({total_gb:.2f} GB)\n")
    
    return filtered_dir

def compress_images(image_dir, quality=85, max_size=600):
    """Compress images to reduce storage."""
    print(f"🗜️  Compressing images (quality={quality}, max {max_size}px)...\n")
    
    total_before = 0
    total_after = 0
    count = 0
    
    for img_path in sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.JPG")):
        try:
            img = Image.open(img_path)
            total_before += img_path.stat().st_size / (1024*1024)
            
            # Resize if necessary
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size*ratio), int(img.size*ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save compressed
            img.save(img_path, "JPEG", quality=quality, optimize=True)
            total_after += img_path.stat().st_size / (1024*1024)
            count += 1
            
            if count % 500 == 0:
                print(f"  Compressed {count} images...")
        
        except Exception as e:
            print(f"  ⚠️  Error: {img_path.name}: {e}")
    
    savings = total_before - total_after
    savings_pct = (100 * savings / total_before) if total_before > 0 else 0
    
    print(f"\n✅ Compression complete:")
    print(f"   Before: {total_before:.1f} GB")
    print(f"   After:  {total_after:.1f} GB")
    print(f"   Saved:  {savings:.1f} GB ({savings_pct:.0f}%)\n")

def main():
    os.chdir(Path(__file__).parent.parent)
    
    try:
        setup_kaggle()
        source_dir = download_plantvillage()
        filtered_dir = filter_by_crop(source_dir, TARGET_CROPS, IMAGES_PER_CLASS)
        compress_images(filtered_dir)
        
        print("=" * 70)
        print("✅ PHASE 0.1 COMPLETE!")
        print("=" * 70)
        print(f"\nDataset location: {filtered_dir}")
        print(f"Next: Run scripts/organize_smart.py\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        exit(1)

if __name__ == "__main__":
    main()