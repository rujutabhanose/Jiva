
#!/usr/bin/env python3
"""
Diagnose nutrient dataset and model issues
Run FIRST before training
"""

import numpy as np
import sys
from pathlib import Path
from collections import Counter

def diagnose():
    print("\n" + "="*80)
    print("NUTRIENT DATASET & MODEL DIAGNOSIS")
    print("="*80)
    
    # Load data
    data_dir = Path("./data/nutrient_training/processed")
    
    try:
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")
    except FileNotFoundError as e:
        print(f"❌ DATA ERROR: {e}")
        return False
    
    print("\n[1] DATA SHAPE CHECK")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Expected: (N, 224, 224, 3) and (N,)")
    
    # Check values
    print("\n[2] DATA VALUE RANGES")
    print(f"  X_train min: {X_train.min()}, max: {X_train.max()}")
    print(f"  Should be: 0-255 or 0.0-1.0")
    if X_train.max() > 2:
        print("  ⚠️  Values in 0-255 range (good)")
    else:
        print("  ⚠️  Values in 0-1 range (also okay)")
    
    print(f"\n  y_train min: {y_train.min()}, max: {y_train.max()}")
    print(f"  y_train unique classes: {len(np.unique(y_train))}")
    
    # Check class distribution
    print("\n[3] CLASS DISTRIBUTION")
    class_counts_train = Counter(y_train)
    class_counts_val = Counter(y_val)
    class_counts_test = Counter(y_test)
    
    print("  Training set:")
    for c in sorted(class_counts_train.keys()):
        count = class_counts_train[c]
        pct = 100 * count / len(y_train)
        print(f"    Class {c}: {count:5d} samples ({pct:5.1f}%)")
    
    print("\n  Validation set:")
    for c in sorted(class_counts_val.keys()):
        count = class_counts_val[c]
        pct = 100 * count / len(y_val)
        print(f"    Class {c}: {count:5d} samples ({pct:5.1f}%)")
    
    # Check for imbalance
    min_class = min(class_counts_train.values())
    max_class = max(class_counts_train.values())
    imbalance_ratio = max_class / min_class
    
    print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print("  ⚠️  IMBALANCED DATA (need weighted loss or resampling)")
    else:
        print("  ✓ Data is relatively balanced")
    
    # Check for NaN/Inf
    print("\n[4] DATA CORRUPTION CHECK")
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  ❌ DATA IS CORRUPTED!")
        return False
    else:
        print("  ✓ Data is clean")
    
    # Recommend model size
    print("\n[5] MODEL RECOMMENDATION")
    total_samples = len(X_train)
    num_classes = len(np.unique(y_train))
    
    print(f"  Total samples: {total_samples}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Samples per class: {total_samples // num_classes}")
    
    if total_samples < 10000:
        print("  ⚠️  Dataset size: SMALL (< 10K)")
        print("  MUST USE: EfficientNetV2-S or MobileNetV3 with ImageNet pretraining")
        print("  DO NOT USE: EfficientNetV2-M (too large)")
    elif total_samples < 50000:
        print("  ✓ Dataset size: MEDIUM (10-50K)")
        print("  USE: MobileNetV3 or EfficientNetV2-S with ImageNet pretraining")
    else:
        print("  ✓ Dataset size: LARGE (> 50K)")
        print("  CAN USE: Any model with proper training strategy")
    
    print("\n[6] CRITICAL CHECKS")
    checks = [
        ("Data shape correct", X_train.shape[0] > 0 and X_train.shape[-1] == 3),
        ("Labels present", len(y_train) == X_train.shape[0]),
        ("No data corruption", nan_count == 0 and inf_count == 0),
        ("Classes present", len(np.unique(y_train)) >= 2),
        ("Enough data per class", total_samples // num_classes >= 50),
    ]
    
    all_pass = True
    for check_name, check_result in checks:
        symbol = "✓" if check_result else "❌"
        print(f"  {symbol} {check_name}")
        if not check_result:
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ DATA LOOKS GOOD - Ready to train")
        print("\nNEXT STEPS:")
        print("  1. Use MobileNetV3 with ImageNet pretraining (NOT EfficientNetV2-M)")
        print("  2. Follow training script with proper transfer learning")
        print("  3. Enable augmentation to boost from 7.6K to ~15K effective samples")
    else:
        print("❌ DATA HAS ISSUES - Fix before training")
    print("="*80 + "\n")
    
    return all_pass

if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)