#!/usr/bin/env python3
"""
Plant species identification using Pl@ntNet API (primary).
Backup: see plant_identifier_huggingface_backup.py
  - Uses juppy44/plant-identification-2m-vit-b via HuggingFace InferenceClient
  - Currently unavailable: model not deployed on HuggingFace serverless inference API
"""

import os
import requests
from typing import Dict
import logging

logger = logging.getLogger(__name__)

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"


class PlantIdentifier:
    """Identifies plant species using Pl@ntNet API."""

    def __init__(self):
        if not PLANTNET_API_KEY:
            raise ValueError("PLANTNET_API_KEY environment variable is required")
        self.active_model = "plantnet-api"
        logger.info("✅ PlantIdentifier initialized with Pl@ntNet API")

    def identify(
        self,
        image_path: str,
        top_k: int = 5,
        region: str = "India",
        context: str = "unknown"
    ) -> Dict:
        """
        Identify plant species from image using Pl@ntNet API.

        Returns: {
            "plant": str,
            "confidence": float (0-1),
            "top_k": [{"plant_name": str, "confidence": float, "confidence_percent": float}],
            "action": "CONFIDENT" | "UNCERTAIN" | "UNKNOWN",
            "reasoning": str,
            "model_used": str
        }
        """
        try:
            with open(image_path, "rb") as f:
                files = [("images", (os.path.basename(image_path), f, "image/jpeg"))]
                response = requests.post(
                    PLANTNET_URL,
                    files=files,
                    params={"api-key": PLANTNET_API_KEY, "nb-results": top_k},
                    data={"organs": ["auto"]},
                    timeout=20
                )

            if response.status_code != 200:
                logger.warning(f"Pl@ntNet API error: {response.status_code} {response.text[:200]}")
                return self._not_found_response()

            results = response.json().get("results", [])
            if not results:
                return self._not_found_response()

            top = results[0]
            top_conf = float(top.get("score", 0))
            species = top.get("species", {})
            top_name = (
                (species.get("commonNames") or [None])[0]
                or species.get("scientificNameWithoutAuthor", "Unknown")
            )

            if top_conf >= 0.75:
                action = "CONFIDENT"
            elif top_conf >= 0.40:
                action = "UNCERTAIN"
            else:
                action = "UNKNOWN"

            formatted_alternatives = []
            for r in results[1:3]:
                r_species = r.get("species", {})
                r_name = (
                    (r_species.get("commonNames") or [None])[0]
                    or r_species.get("scientificNameWithoutAuthor", "Unknown")
                )
                formatted_alternatives.append({
                    "plant_name": r_name,
                    "confidence": float(r.get("score", 0)),
                    "confidence_percent": float(r.get("score", 0)) * 100
                })

            return {
                "plant": top_name,
                "scientific_name": species.get("scientificNameWithoutAuthor"),
                "family": species.get("family", {}).get("scientificNameWithoutAuthor"),
                "confidence": top_conf,
                "top_k": formatted_alternatives,
                "action": action,
                "reasoning": f"Pl@ntNet confidence: {top_conf*100:.0f}% ({top_name})",
                "model_used": self.active_model
            }

        except Exception as e:
            logger.error(f"Pl@ntNet identification failed: {e}")
            return self._not_found_response()

    def _not_found_response(self) -> Dict:
        return {
            "plant": "Unknown",
            "confidence": 0.0,
            "top_k": [],
            "action": "UNKNOWN",
            "reasoning": "Unable to identify from image. Try a clearer photo of the leaf or flower.",
            "model_used": self.active_model
        }


# Global instance
identifier = PlantIdentifier()


def identify_plant(image_path: str, region: str = "India") -> Dict:
    """Convenience function."""
    return identifier.identify(image_path, region=region)
