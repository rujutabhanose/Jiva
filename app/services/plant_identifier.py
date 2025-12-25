from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from typing import List, Dict
import logging
from app.services.plant_info import enrich_with_usda

logger = logging.getLogger(__name__)

MODEL_ID = "juppy44/plant-identification-2m-vit-b"

# Global variables for model and processor
processor = None
model = None


def load_model():
    """
    Load the plant identification model.
    This is called once at startup via main.py lifespan.
    """
    global processor, model

    if processor is not None and model is not None:
        logger.info("Model already loaded, skipping...")
        return

    try:
        logger.info(f"Loading plant identification model: {MODEL_ID}")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        model.eval()
        logger.info(f"✅ Model loaded successfully! Can identify {len(model.config.id2label)} species")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def identify_plant(image_path: str, top_k: int = 3) -> Dict[str, any]:
    """
    Identify a plant from an image and enrich top prediction with USDA data.

    Returns a dict with:
      - primary: enriched top prediction (commonName, scientificName, family, confidence)
      - alternatives: list of other raw predictions
    """
    if processor is None or model is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    try:
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"Processing image: {image_path} (size: {image.size})")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits.softmax(dim=-1)[0]
        topk = torch.topk(probs, k=min(top_k, len(model.config.id2label)))

        raw_results = []
        for score, idx in zip(topk.values, topk.indices):
            label = model.config.id2label[idx.item()]  # e.g. "Abies alba"
            confidence = float(score)
            raw_results.append({
                "plant_name": label,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2)
            })

        if not raw_results:
            raise RuntimeError("No predictions returned from plant ID model")

        # Take top prediction and enrich with USDA info
        top = raw_results[0]
        sci_name = top["plant_name"]
        primary = enrich_with_usda(sci_name, top["confidence"])

        logger.info(
            f"Top prediction: {primary.get('commonName') or sci_name} "
            f"({primary['confidence']}%), family={primary.get('family')}"
        )

        return {
            "primary": primary,
            "alternatives": raw_results[1:]
        }

    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        raise ValueError(f"Image file not found: {image_path}")
    except Exception as e:
        logger.error(f"Error during plant identification: {e}")
        raise RuntimeError(f"Failed to identify plant: {str(e)}")