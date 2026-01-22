#!/usr/bin/env python3
"""
Plant species identification using juppy44/plant-identification-2m-vit-b
Pretrained on 2M+ iNaturalist images
Coverage: ~10,000 plant species globally
"""

import json
from pathlib import Path
from typing import List, Dict
from huggingface_hub import InferenceClient

# HuggingFace model
MODEL_NAME = "juppy44/plant-identification-2m-vit-b"
HF_TOKEN = None  # Optional: set HUGGINGFACE_TOKEN env var for higher rate limits

class PlantIdentifier:
    """Identifies plant species from leaf/flower images."""

    def __init__(self):
        # Use new router endpoint (api-inference.huggingface.co is deprecated)
        self.client = InferenceClient(
            model=f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
        )

        # Load plant knowledge base (try both .json and no extension)
        kb_path = Path(__file__).parent / "plant_knowledge_base.json"
        if not kb_path.exists():
            kb_path = Path(__file__).parent / "plant_knowledge_base"

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
            "reasoning": str
        }
        """
        
        # Get predictions (pass file path directly - InferenceClient expects path, URL, or binary)
        predictions = self.client.image_classification(image=image_path)
        
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
        
        return {
            "plant": top_species,
            "confidence": float(top_conf),
            "top_k": [
                {"species": s["label"], "score": float(s["score"])}
                for s in scored[:3]
            ],
            "action": action,
            "reasoning": f"Model confidence: {top_conf*100:.0f}% ({top_species})"
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
            "reasoning": "Unable to identify from image. Try a clearer photo of the leaf or flower."
        }

# Global instance
identifier = PlantIdentifier()

def identify_plant(image_path: str, region: str = "India") -> Dict:
    """Convenience function."""
    return identifier.identify(image_path, region=region)