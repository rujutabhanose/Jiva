#!/usr/bin/env python3
"""
Extract disease class names and create mapping JSON
"""

import json
from pathlib import Path

DATA_DIR = Path("data/organized")
BACKEND_DIR = Path("backend")

def extract_classes():
    """Get all class names from train directory."""
    print("📋 Extracting class names...\n")
    
    BACKEND_DIR.mkdir(exist_ok=True)
    
    classes = sorted([
        d.name for d in (DATA_DIR / "train").iterdir() 
        if d.is_dir()
    ])
    
    mapping = {
        "num_classes": len(classes),
        "classes": classes,
        "index_to_name": {str(i): name for i, name in enumerate(classes)},
        "name_to_index": {name: str(i) for i, name in enumerate(classes)}
    }
    
    # Save
    out_path = BACKEND_DIR / "disease_class_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Extracted {len(classes)} disease classes\n")
    print("Classes:")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")
    
    print(f"\n✅ Saved to: {out_path}\n")
    return mapping

if __name__ == "__main__":
    extract_classes()