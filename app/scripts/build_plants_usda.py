"""
Build plants_usda.json from USDA_PLANTS.txt

Input:
  backend/data/USDA_PLANTS.txt  (USDA export with header:
    "Symbol","Synonym Symbol","Scientific Name with Author","Common Name","Family")

Output:
  backend/data/plants_usda.json
"""

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]        # backend/
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = DATA_DIR / "USDA_PLANTS.txt"
OUTPUT_FILE = DATA_DIR / "plants_usda.json"


def normalize_key(scientific_name_with_author: str) -> str:
    """
    Turn 'Abies alba Mill.' → 'abies_alba'
    """
    sci = scientific_name_with_author.split("(")[0].strip()
    parts = sci.split()
    if len(parts) >= 2:
        key = f"{parts[0]}_{parts[1]}"
    else:
        key = sci.replace(" ", "_")
    return key.lower()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"USDA file not found at {INPUT_FILE}")

    plants = {}

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("Symbol") or "").strip()
            sci = (row.get("Scientific Name with Author") or "").strip()
            common = (row.get("Common Name") or "").strip()
            family = (row.get("Family") or "").strip()

            if not sci:
                continue

            key = normalize_key(sci)

            # Prefer entries that actually have common name / family
            score = (1 if common else 0) + (1 if family else 0)
            existing = plants.get(key)
            if existing:
                existing_score = (
                    1 if existing.get("commonName") else 0
                ) + (1 if existing.get("family") else 0)
                if score <= existing_score:
                    continue

            plants[key] = {
                "symbol": symbol or None,
                "commonName": common or None,
                "scientificName": sci,
                "family": family or None,
            }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(plants, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(plants)} plants to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()