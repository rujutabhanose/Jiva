"""
Check all images are valid and not corrupted
"""

from pathlib import Path
from PIL import Image
import json

DATA_DIR = Path(__file__).parent.parent / "data" / "organized"

def verify_dataset():
    """Verify all images are loadable."""
    print("🔍 Verifying dataset integrity...\n")
    
    errors = []
    stats = {}
    
    for split in ["train", "val", "test", "calibration"]:
        split_dir = DATA_DIR / split
        count = 0
        invalid = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            # Check both .jpg and .JPG extensions
            for pattern in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                for img_path in sorted(class_dir.glob(pattern)):
                    count += 1
                    try:
                        img = Image.open(img_path)
                        img.verify()
                    except Exception as e:
                        invalid += 1
                        errors.append(f"{img_path}: {e}")
                        try:
                            img_path.unlink()
                            print(f"  ❌ Removed corrupt: {img_path.name}")
                        except:
                            pass
        
        stats[split] = {
            "total": count,
            "invalid": invalid,
            "valid": count - invalid
        }
        print(f"  {split.upper()}: {count} images ({count - invalid} valid)")
    
    print("\n" + "=" * 70)
    if errors:
        print(f"⚠️  Found {len(errors)} corrupted images (removed)")
    else:
        print(f"✅ All images valid!")
    print("=" * 70 + "\n")
    
    return stats

if __name__ == "__main__":
    verify_dataset()