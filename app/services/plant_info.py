import json
from pathlib import Path
from typing import Dict, Any

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "plants_usda.json"

with DATA_FILE.open("r", encoding="utf-8") as f:
    USDA_PLANTS: Dict[str, Any] = json.load(f)


def _normalize_sci_key(scientific_name: str) -> str:
    sci = scientific_name.split("(")[0].strip()
    parts = sci.split()
    if len(parts) >= 2:
        key = f"{parts[0]}_{parts[1]}"
    else:
        key = sci.replace(" ", "_")
    return key.lower()


def enrich_with_usda(scientific_name: str, confidence: float) -> Dict[str, Any]:
    key = _normalize_sci_key(scientific_name)
    info = USDA_PLANTS.get(key, {})

    return {
        "commonName": info.get("commonName") or scientific_name,
        "scientificName": scientific_name,
        "family": info.get("family"),
        "usdaSymbol": info.get("symbol"),
        "confidence": round(confidence * 100, 1),
    }