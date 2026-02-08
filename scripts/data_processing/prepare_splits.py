#!/usr/bin/env python3
"""
Create stratified train/val/test splits with correct class_id.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("./data/processed")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def create_splits():
    print("JIVA PLANTS: FIXED TRAIN/VAL/TEST SPLITS")
    print("=" * 60)

    unified_index_path = PROCESSED_DIR / "unified_image_index.json"
    if not unified_index_path.exists():
        raise FileNotFoundError(f"{unified_index_path} not found. Run data_processing_fixed.py first.")

    with open(unified_index_path) as f:
        all_images = json.load(f)

    # Build global mapping class_name -> id
    class_name_to_id = {}
    next_id = 0
    for img in all_images:
        cname = img["class"]
        if cname not in class_name_to_id:
            class_name_to_id[cname] = next_id
            next_id += 1

    print(f"Total unique classes: {len(class_name_to_id)}")
    print("Example mapping:")
    for cname, cid in list(class_name_to_id.items())[:10]:
        print(f"  {cid:2d} -> {cname}")

    # Attach correct class_id to each image
    by_class = defaultdict(list)
    for img in all_images:
        cname = img["class"]
        cid = class_name_to_id[cname]
        img["class_id"] = int(cid)  # overwrite / ensure correct
        by_class[cname].append(img)

    train_images, val_images, test_images = [], [], []

    for class_name, images in by_class.items():
        n = len(images)
        indices = np.arange(n)

        train_idx, temp_idx = train_test_split(
            indices,
            train_size=TRAIN_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        for idx in train_idx:
            img = images[idx].copy()
            img["split"] = "train"
            train_images.append(img)

        for idx in val_idx:
            img = images[idx].copy()
            img["split"] = "val"
            val_images.append(img)

        for idx in test_idx:
            img = images[idx].copy()
            img["split"] = "test"
            test_images.append(img)

        print(
            f"{class_name:35s}: "
            f"{len(train_idx):4d} train, "
            f"{len(val_idx):4d} val, "
            f"{len(test_idx):4d} test"
        )

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
        "metadata": {
            "total_images": len(all_images),
            "total_classes": len(class_name_to_id),
        },
    }

    with open(PROCESSED_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    with open(PROCESSED_DIR / "class_name_to_id.json", "w") as f:
        json.dump(class_name_to_id, f, indent=2)

    print("\nSplits created.")
    print(f"  Train: {len(train_images)} ({100 * len(train_images) / len(all_images):.1f}%)")
    print(f"  Val:   {len(val_images)} ({100 * len(val_images) / len(all_images):.1f}%)")
    print(f"  Test:  {len(test_images)} ({100 * len(test_images) / len(all_images):.1f}%)")

if __name__ == "__main__":
    create_splits()