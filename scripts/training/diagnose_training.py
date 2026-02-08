#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
import tensorflow as tf

PROCESSED_DIR = Path("./data/processed")

print("DIAGNOSING JIVA PLANTS TRAINING ISSUE...\n")

with open(PROCESSED_DIR / "splits.json") as f:
    splits = json.load(f)

train_data = splits["train"]
val_data = splits["val"]

print(f"✓ Train samples: {len(train_data)}")
print(f"✓ Val samples:   {len(val_data)}")

classes = Counter([img["class"] for img in train_data])
print(f"✓ Unique classes: {len(classes)}")

print("\nTop 5 classes:")
for cls, count in classes.most_common(5):
    print(f"  {cls}: {count}")

print("\nBottom 5 classes:")
for cls, count in sorted(classes.items(), key=lambda x: x[1])[:5]:
    print(f"  {cls}: {count}")

class_ids = set([img["class_id"] for img in train_data])
print(f"\nClass ID range: {min(class_ids)} to {max(class_ids)}")

# Test image path
sample = train_data[0]
img_path = PROCESSED_DIR / sample["path"]
if img_path.exists():
    print(f"✓ Image exists: {img_path}")
else:
    print(f"✗ IMAGE NOT FOUND: {img_path}")

print("\nDiagnostic complete!")