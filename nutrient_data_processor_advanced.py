"""
Advanced Nutrient Dataset Processor
- Loads multiple data sources
- Creates train/val/test splits
- Aggressive augmentation for small datasets
- Handles imbalanced classes
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'raw_data_path': './data/nutrient_training/raw',
    'processed_data_path': './data/nutrient_training/processed',
    'image_size': (224, 224),
    'augmentation_factor': 5,  # Multiply dataset by 5x
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'classes': ['N', 'P', 'K', 'Mg', 'Ca', 'Fe', 'Mn', 'B', 'Healthy'],
}

# ============================================================================
# PART 1: IMAGE PREPROCESSING (Universal for any plant)
# ============================================================================

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Universal image preprocessing for ANY plant/leaf
    - Works for coffee, tomato, corn, wheat, rice, etc.
    - Normalizes for different lighting, angles, backgrounds
    """
    print(f"  Loading: {Path(image_path).name}")
    
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"    ⚠️ Failed to load {image_path}")
            return None
        
        original_shape = image.shape
        
        # Step 1: Color correction (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Resize to target (preserve aspect ratio with padding)
        h, w = image.shape[:2]
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad to exact size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        image = cv2.copyMakeBorder(
            image, pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_REFLECT
        )
        
        # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Enhances leaf details regardless of lighting
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        
        # Step 4: Normalize color
        # Removes shadows and uneven lighting
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Step 5: Sharpen (enhance leaf texture)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        image = cv2.filter2D(image, -1, kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
        
    except Exception as e:
        print(f"    ❌ Error processing {image_path}: {e}")
        return None

# ============================================================================
# PART 2: DATA AUGMENTATION (Creates more training data)
# ============================================================================

def augment_image(image, num_variations=5):
    """
    Generate augmented versions of single image
    - Rotation, flip, brightness, contrast
    - Creates synthetic diversity
    """
    augmented = [image.copy()]  # Original
    
    for i in range(num_variations - 1):
        aug_image = image.copy()
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        h, w = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h))
        
        # Random horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            aug_image = cv2.flip(aug_image, 1)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        aug_image = np.clip(aug_image * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        contrast = np.random.uniform(0.8, 1.2)
        aug_image = cv2.convertScaleAbs(aug_image, alpha=contrast, beta=0)
        
        # Random Gaussian noise (5% chance)
        if np.random.rand() > 0.95:
            noise = np.random.normal(0, 10, aug_image.shape).astype(np.uint8)
            aug_image = cv2.add(aug_image, noise)
        
        # Random blur (10% chance, to simulate motion)
        if np.random.rand() > 0.9:
            aug_image = cv2.GaussianBlur(aug_image, (5, 5), 0)
        
        augmented.append(aug_image)
    
    return augmented

# ============================================================================
# PART 3: DATA LOADING & PROCESSING
# ============================================================================

def load_and_process_data():
    """
    Main data loading pipeline
    """
    print("\n" + "="*70)
    print("NUTRIENT DATASET PROCESSOR - ADVANCED")
    print("="*70)
    
    # Step 1: Load raw data
    print("\n LOADING RAW DATA...")
    print(f"    Source: {CONFIG['raw_data_path']}")
    
    all_images = []
    all_labels = []
    class_counts = {}
    
    for class_name in CONFIG['classes']:
        class_path = Path(CONFIG['raw_data_path']) / class_name
        if not class_path.exists():
            print(f"    ⚠️  {class_name}: NOT FOUND")
            continue
        
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        print(f"    ✓ {class_name:10s}: {len(images):3d} images")
        class_counts[class_name] = len(images)
        
        for img_path in images:
            preprocessed = preprocess_image(str(img_path))
            if preprocessed is not None:
                all_images.append(preprocessed)
                all_labels.append(class_name)
    
    if len(all_images) == 0:
        print("\n    ❌ ERROR: No images found!")
        print(f"    Check: {CONFIG['raw_data_path']}")
        return None
    
    print(f"\n    Total images loaded: {len(all_images)}")
    print(f"    Unique classes: {len(set(all_labels))}")
    
    # Step 2: Data augmentation
    print("\n AUGMENTING DATA (Creating synthetic variations)...")
    print(f"    Multiplication factor: {CONFIG['augmentation_factor']}x")
    
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(all_images, all_labels):
        variations = augment_image(image, CONFIG['augmentation_factor'])
        augmented_images.extend(variations)
        augmented_labels.extend([label] * len(variations))
    
    print(f"    ✓ Original images: {len(all_images)}")
    print(f"    ✓ After augmentation: {len(augmented_images)}")
    print(f"    ✓ Augmentation ratio: {len(augmented_images) / len(all_images):.1f}x")
    
    # Step 3: Handle class imbalance
    print("\n HANDLING CLASS IMBALANCE...")
    
    # Count each class
    unique, counts = np.unique(augmented_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    print("    Distribution before balancing:")
    for class_name, count in sorted(class_distribution.items()):
        percentage = (count / len(augmented_labels)) * 100
        print(f"      {class_name:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # Oversample minority classes
    max_count = max(counts)
    balanced_images = []
    balanced_labels = []
    
    for class_name in CONFIG['classes']:
        class_indices = [i for i, label in enumerate(augmented_labels) if label == class_name]
        if not class_indices:
            continue
        
        current_count = len(class_indices)
        if current_count < max_count:
            # Oversample
            ratio = max_count / current_count
            new_indices = np.random.choice(class_indices, int(current_count * ratio), replace=True)
        else:
            new_indices = class_indices
        
        for idx in new_indices:
            balanced_images.append(augmented_images[idx])
            balanced_labels.append(augmented_labels[idx])
    
    print(f"\n    ✓ After balancing: {len(balanced_images)} images")
    print("    Distribution after balancing:")
    unique_balanced, counts_balanced = np.unique(balanced_labels, return_counts=True)
    for class_name, count in zip(unique_balanced, counts_balanced):
        percentage = (count / len(balanced_labels)) * 100
        print(f"      {class_name:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # Step 4: Train/Val/Test split
    print("\n SPLITTING DATA...")
    print(f"    Train: {CONFIG['train_split']*100:.0f}%")
    print(f"    Val:   {CONFIG['val_split']*100:.0f}%")
    print(f"    Test:  {CONFIG['test_split']*100:.0f}%")
    
    # Convert to numpy
    X = np.array(balanced_images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y_encoded = LabelEncoder().fit_transform(balanced_labels)
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded,
        test_size=CONFIG['test_split'],
        random_state=42,
        stratify=y_encoded
    )
    
    val_size = CONFIG['val_split'] / (CONFIG['train_split'] + CONFIG['val_split'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n    ✓ Train: {len(X_train)} images")
    print(f"    ✓ Val:   {len(X_val)} images")
    print(f"    ✓ Test:  {len(X_test)} images")
    
    # Step 5: Save processed data
    print("\n SAVING PROCESSED DATA...")
    processed_path = Path(CONFIG['processed_data_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    np.save(str(processed_path / 'X_train.npy'), X_train)
    np.save(str(processed_path / 'X_val.npy'), X_val)
    np.save(str(processed_path / 'X_test.npy'), X_test)
    np.save(str(processed_path / 'y_train.npy'), y_train)
    np.save(str(processed_path / 'y_val.npy'), y_val)
    np.save(str(processed_path / 'y_test.npy'), y_test)
    
    # Save metadata
    class_list = CONFIG['classes']
    metadata = {
        'classes': class_list,
        'num_classes': len(class_list),
        'class_to_id': {cls: idx for idx, cls in enumerate(class_list)},
        'id_to_class': {idx: cls for idx, cls in enumerate(class_list)},
        'image_size': CONFIG['image_size'],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }
    
    with open(str(processed_path / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Data saved to: {processed_path}")
    print(f"    ✓ Metadata saved")
    
    print("\n" + "="*70)
    print("✅ DATA PROCESSING COMPLETE")
    print("="*70)
    
    return metadata

# ============================================================================
# PART 4: VERIFY DATA
# ============================================================================

def verify_processed_data():
    """
    Verify saved data is correct
    """
    print("\n[VERIFICATION]")
    processed_path = Path(CONFIG['processed_data_path'])
    
    try:
        X_train = np.load(str(processed_path / 'X_train.npy'))
        X_val = np.load(str(processed_path / 'X_val.npy'))
        X_test = np.load(str(processed_path / 'X_test.npy'))
        y_train = np.load(str(processed_path / 'y_train.npy'))
        
        print(f"✓ X_train shape: {X_train.shape} (images: {X_train.shape})")
        print(f"✓ X_val shape:   {X_val.shape} (images: {X_val.shape})")
        print(f"✓ X_test shape:  {X_test.shape} (images: {X_test.shape})")
        print(f"✓ Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"✓ Classes in training: {len(np.unique(y_train))}")
        
        with open(str(processed_path / 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded: {metadata['num_classes']} classes")
        
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    metadata = load_and_process_data()
    if metadata:
        verify_processed_data()