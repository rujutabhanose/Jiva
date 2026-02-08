#!/usr/bin/env python3
"""
Data processing and normalization
- Resize all images to 224x224
- Create class mappings
- Generate dataset indexes
- Create augmentation configurations
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

# Configuration
RAW_DATA_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
IMG_SIZE = 224
NUM_WORKERS = mp.cpu_count()

# Disease class mapping (unified across datasets)
DISEASE_MAPPING = {
    # Apple
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "apple_black_rot",
    "Apple___Cedar_apple_rust": "apple_cedar_rust",
    "Apple___healthy": "apple_healthy",
    
    # Blueberry
    "Blueberry___healthy": "blueberry_healthy",
    
    # Cherry
    "Cherry___healthy": "cherry_healthy",
    "Cherry___Powdery_mildew": "cherry_powdery_mildew",
    
    # Corn
    "Corn___Cercospora_leaf_spot": "corn_cercospora",
    "Corn___Common_rust": "corn_common_rust",
    "Corn___Northern_Leaf_Blight": "corn_northern_blight",
    "Corn___healthy": "corn_healthy",
    
    # Grape
    "Grape___Black_rot": "grape_black_rot",
    "Grape___Esca_(Black_Measles)": "grape_esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
    "Grape___healthy": "grape_healthy",
    "grape_downy_mildew": "grape_downy_mildew",  # Niphad dataset
    "grape_powdery_mildew": "grape_powdery_mildew",  # Niphad dataset
    "grape_bacterial_spot": "grape_bacterial_spot",  # Niphad dataset
    
    # Peach
    "Peach___Bacterial_shot_hole": "peach_bacterial_shot",
    "Peach___healthy": "peach_healthy",
    
    # Pepper
    "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper,_bell___healthy": "pepper_healthy",
    
    # Potato
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
    
    # Strawberry
    "Strawberry___Leaf_scorch": "strawberry_leaf_scorch",
    "Strawberry___healthy": "strawberry_healthy",
    
    # Tomato
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellowing",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic",
    "Tomato___healthy": "tomato_healthy",
}

def normalize_image(image_path):
    """Load and normalize image to 224x224"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(dataset_name, source_path):
    """Process a single dataset"""
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}")
    print(f"{'='*70}")
    
    output_path = PROCESSED_DIR / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_index = {}
    image_index = []
    class_counter = 0
    image_counter = 0
    
    # Get all classes
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name_raw = class_dir.name
        
        # Normalize class name
        class_name = DISEASE_MAPPING.get(class_name_raw, class_name_raw.lower())
        
        if class_name not in class_index:
            class_index[class_name] = class_counter
            class_counter += 1
        
        class_id = class_index[class_name]
        
        # Create class directory in output
        class_output = output_path / class_name
        class_output.mkdir(exist_ok=True)
        
        # Process all images in this class
        image_paths = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.JPG")) + \
                      list(class_dir.glob("*.png")) + \
                      list(class_dir.glob("*.PNG"))
        
        for img_path in tqdm(image_paths, desc=f"{class_name}", leave=False):
            try:
                img_array = normalize_image(img_path)
                if img_array is not None:
                    # Save normalized image
                    output_img_path = class_output / f"{image_counter:06d}.jpg"
                    Image.fromarray(img_array).save(output_img_path, quality=95)
                    
                    image_index.append({
                        "id": image_counter,
                        "class": class_name,
                        "class_id": class_id,
                        "path": f"{class_name}/{image_counter:06d}.jpg",
                        "source_dataset": dataset_name
                    })
                    image_counter += 1
            except Exception as e:
                print(f"Error with {img_path}: {e}")
    
    # Save class index
    with open(output_path / "class_index.json", "w") as f:
        json.dump(class_index, f, indent=2)
    
    # Save image index
    with open(output_path / "image_index.json", "w") as f:
        json.dump(image_index, f)
    
    print(f"\nDataset {dataset_name} processed:")
    print(f"  - Classes: {len(class_index)}")
    print(f"  - Total images: {len(image_index)}")
    print(f"  - Output: {output_path}")
    
    return class_index, image_index

# Main execution
if __name__ == "__main__":
    print("JIVA PLANTS: DATA PROCESSING")
    print("="*70)
    
    all_classes = {}
    all_images = []
    
    # Process each dataset
    datasets_to_process = [
        ("plant_village", RAW_DATA_DIR / "plant-village" / "segmented"),
        ("plantify_dr", RAW_DATA_DIR / "plantify-dr"),
        ("niphad_grape", RAW_DATA_DIR / "niphad-grape"),
    ]
    
    for dataset_name, source_path in datasets_to_process:
        if source_path.exists():
            classes, images = process_dataset(dataset_name, source_path)
            all_classes.update(classes)
            all_images.extend(images)
        else:
            print(f"⚠ Dataset not found: {source_path}")
    
    # Create unified class mapping
    unified_classes = {v: k for k, v in sorted(all_classes.items(), key=lambda x: x)}
    with open(PROCESSED_DIR / "unified_class_index.json", "w") as f:
        json.dump(unified_classes, f, indent=2)
    
    # Create unified image index
    with open(PROCESSED_DIR / "unified_image_index.json", "w") as f:
        json.dump(all_images, f)
    
    print("\n" + "="*70)
    print(f"FINAL STATS:")
    print(f"  - Total Classes: {len(unified_classes)}")
    print(f"  - Total Images: {len(all_images)}")
    print(f"  - Output Dir: {PROCESSED_DIR}")
    print("="*70)
    print("\nNEXT STEP: Run prepare_splits.py")