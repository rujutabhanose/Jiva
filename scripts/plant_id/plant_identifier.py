#!/usr/bin/env python3
"""
Plant species identification using juppy44/plant-identification-2m-vit-b
Pretrained on 2M+ iNaturalist images
Coverage: ~14,000 plant species globally
"""

import json
from pathlib import Path
from typing import List, Dict
import logging
from huggingface_hub import InferenceClient

# HuggingFace model
MODEL_NAME = "juppy44/plant-identification-2m-vit-b"  # 14k species, 2M+ training images

# HuggingFace authentication token (required for gated models like juppy44)
# Token must have "Make calls to serverless Inference API" permission
# Set HUGGINGFACE_TOKEN environment variable before running
import os
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is required")

logger = logging.getLogger(__name__)

class PlantIdentifier:
    """Identifies plant species from leaf/flower images."""

    def __init__(self):
        # Initialize HuggingFace Inference Client with authentication
        try:
            logger.info(f"Initializing model: {MODEL_NAME}")
            # Use token for authentication (required for gated model)
            self.client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
            self.active_model = MODEL_NAME
            logger.info(f"✅ Successfully initialized model: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Unable to initialize plant identification model: {e}")

        # Load plant knowledge base from data directory
        backend_root = Path(__file__).resolve().parents[2]  # .../backend
        kb_path = backend_root / "data" / "knowledge_bases" / "plant_knowledge_base.json"
        if not kb_path.exists():
            kb_path = backend_root / "data" / "knowledge_bases" / "plant_knowledge_base"

        if kb_path.exists():
            with open(kb_path) as f:
                self.kb = json.load(f)
        else:
            self.kb = {"common_species": {}}
    
    def identify(
        self,
        image_path: str,
        top_k: int = 5,
        region: str = "India",
        context: str = "unknown"
    ) -> Dict:
        """
        Identify plant species from image.

        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            region: Geographic region (for re-ranking)
            context: "houseplant", "outdoor", "crop", "wild"

        Returns: {
            "plant": str,
            "confidence": float (0-1),
            "top_k": [{"species": str, "score": float}],
            "action": "CONFIDENT" | "UNCERTAIN" | "UNKNOWN",
            "reasoning": str,
            "model_used": str
        }
        """

        # Get predictions from HuggingFace Inference API
        try:
            predictions = self.client.image_classification(image=image_path)
        except Exception:
            return self._not_found_response()
        
        # Extract top predictions
        top_preds = predictions[:top_k]
        
        if not top_preds:
            return self._not_found_response()
        
        # Apply re-ranking heuristics
        scored = self._apply_region_bias(top_preds, region, context)
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)
        
        # Decision logic
        top_species = scored[0]["label"]
        top_conf = scored[0]["score"]
        
        if top_conf >= 0.75:
            action = "CONFIDENT"
        elif top_conf >= 0.50:
            action = "UNCERTAIN"
        else:
            action = "UNKNOWN"
        
        # Format alternatives to match expected API schema
        formatted_alternatives = [
            {
                "plant_name": s["label"],
                "confidence": float(s["score"]),
                "confidence_percent": float(s["score"]) * 100
            }
            for s in scored[1:3]  # Skip first (that's primary), take next 2
        ]

        return {
            "plant": top_species,
            "confidence": float(top_conf),
            "top_k": formatted_alternatives,
            "action": action,
            "reasoning": f"Model confidence: {top_conf*100:.0f}% ({top_species})",
            "model_used": self.active_model
        }
    
    def _apply_region_bias(self, predictions: List, region: str, context: str) -> List:
        """
        Re-rank predictions based on region and context.
        E.g., downweight tropical species if in temperate region.
        """
        scored = []
        
        for pred in predictions:
            label = pred["label"]
            score = pred["score"]
            
            # Get species info from KB
            info = self.kb.get("common_species", {}).get(label, {})
            
            # Apply biases
            if region == "India" and info.get("regions"):
                if "South Asia" in info["regions"] or "Tropical" in info["regions"]:
                    score *= 1.2  # Boost common Indian plants
                else:
                    score *= 0.8  # Reduce less common
            
            if context == "houseplant" and info.get("is_houseplant"):
                score *= 1.1
            elif context == "crop" and info.get("is_crop"):
                score *= 1.1
            
            # Clip to [0, 1]
            score = min(1.0, score)
            
            scored.append({**pred, "score": score})
        
        return scored
    
    def _not_found_response(self) -> Dict:
        """Return response for unidentified plant."""
        return {
            "plant": "Unknown",
            "confidence": 0.0,
            "top_k": [],
            "action": "UNKNOWN",
            "reasoning": "Unable to identify from image. Try a clearer photo of the leaf or flower.",
            "model_used": self.active_model or "none"
        }

# Global instance
identifier = PlantIdentifier()

def identify_plant(image_path: str, region: str = "India") -> Dict:
    """Convenience function."""
    return identifier.identify(image_path, region=region)