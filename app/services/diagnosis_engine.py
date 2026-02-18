"""
Hybrid Plant Diagnosis Engine v2.0
Combines TFLite/MobileNetV2 (diseases) + CoLeaf (nutrients) + Rule heuristics

Model loading priority for disease detection:
  1. TFLite quantized (disease_detector_int8.tflite) - fastest, mobile-compatible
  2. TFLite full (disease_detector.tflite) - fallback
  3. HuggingFace pipeline - cloud fallback

With continuous learning support:
- Tracks model versions for each prediction
- Results include model_version_id for feedback correlation
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
import asyncio

import numpy as np
from PIL import Image
from app.services.tflite_compat import TFLiteInterpreter, HAS_TFLITE
from huggingface_hub import InferenceClient

from app.services.coleaf_engine import run_coleaf, get_model_version_id as get_coleaf_version
from .disease_mapping import (
    get_diagnosis_info, normalize_label, is_nutrient_deficiency,
    is_fungal_disease, is_bacterial_disease, get_nutrient_type
)
from .hybrid_plant_identifier import identify_plant_hybrid

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]  # .../backend
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

TFLITE_QUANTIZED = MODELS_DIR / "disease_detector_int8.tflite"
TFLITE_FULL = MODELS_DIR / "disease_detector.tflite"
CLASS_INDEX_FILE = DATA_DIR / "unified_class_index.json"

IMG_SIZE = 224

# Load disease class indices
DISEASE_CLASS_NAMES: Dict[int, str] = {}
try:
    with CLASS_INDEX_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)
        DISEASE_CLASS_NAMES = {int(k): v for k, v in raw.items()}
    logger.info(f"Disease class index loaded: {len(DISEASE_CLASS_NAMES)} classes")
except Exception as e:
    logger.warning(f"Failed to load disease class index: {e}")

# TFLite interpreter (loaded lazily)
_tflite_interpreter = None
_tflite_model_path = None
_huggingface_pipeline = None
_disease_model_format = None  # "tflite" or "huggingface"


def _load_tflite_disease_model():
    """Load TFLite disease model with fallback between quantized and full."""
    global _tflite_interpreter, _tflite_model_path, _disease_model_format

    if _tflite_interpreter is not None:
        return _tflite_interpreter

    if not HAS_TFLITE:
        logger.warning("No TFLite runtime available for disease detection")
        return None

    for model_path in [TFLITE_QUANTIZED, TFLITE_FULL]:
        if model_path.exists():
            try:
                interpreter = TFLiteInterpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                _tflite_interpreter = interpreter
                _tflite_model_path = model_path
                _disease_model_format = "tflite"
                logger.info(f"Disease TFLite model loaded: {model_path.name}")
                return interpreter
            except Exception as e:
                logger.warning(f"Failed to load TFLite {model_path.name}: {e}")
                continue

    return None


def _load_huggingface_disease_model():
    """Load HuggingFace pipeline as fallback."""
    global _huggingface_pipeline, _disease_model_format

    if _huggingface_pipeline is not None:
        return _huggingface_pipeline

    try:
        from transformers import pipeline
        _huggingface_pipeline = pipeline(
            "image-classification",
            model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )
        _disease_model_format = "huggingface"
        logger.info("Disease HuggingFace pipeline loaded as fallback")
        return _huggingface_pipeline
    except Exception as e:
        logger.error(f"Failed to load HuggingFace disease pipeline: {e}")
        return None


# Initialize disease model: Try TFLite first, then HuggingFace
_load_tflite_disease_model() or _load_huggingface_disease_model()
if _disease_model_format:
    logger.info(f"Disease detection initialized with {_disease_model_format} model")
else:
    logger.error("Disease detection: No model loaded!")

# LLaVA client for symptom analysis (nutrient deficiencies + broad coverage)
llava_client = InferenceClient(
    model="YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
)

from app.services.quality_gate import image_quality_check


def _identify_plant(image_path: str) -> Optional[str]:
    """Identify the plant species from the image."""
    try:
        # identify_plant_hybrid is async; run it without interfering with any
        # existing event loop. We execute asyncio.run in a separate thread so
        # it never collides with uvicorn/asyncio main loop. Apply a short
        # timeout so slow cloud calls don't block the request for too long.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FUTEX

        def _run():
            return asyncio.run(identify_plant_hybrid(image_path))

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run)
            try:
                result = future.result(timeout=6)  # seconds
            except FUTEX:
                logger.warning("Plant identification timed out (6s). Continuing without plant name.")
                return None

        if result and result.get("primary"):
            plant_name = result["primary"].get("commonName") or result["primary"].get("scientificName")
            logger.info(f"Plant identified: {plant_name} (confidence: {result['primary'].get('confidence', 0):.1f}%)")
            return plant_name

    except Exception as e:
        logger.warning(f"Plant identification failed: {e}. Continuing with diagnosis only.")

    return None


def diagnose_image(image_path: str, top_k: int = 5) -> Dict:
    if not image_quality_check(image_path):
        return {
            "success": False,
            "reason": "Image too blurry or unclear. Please take a closer photo of a leaf.",
            "diagnoses": [],
            "plant_health_score": None,
            "plant_name": None
        }
    """Hybrid diagnosis: MobileNetV2 (diseases) + CoLeaf (nutrients) + Plant Identification"""
    # Disease model (PlantVillage)
    disease_results = _run_disease_model(image_path, top_k)

    # Nutrient model (CoLeaf)
    nutrient_results = run_coleaf(image_path)

    # If LLaVA is still off, keep empty symptom_results
    symptom_results = {"diagnoses": [], "confidence": 0.0}
    # symptom_results = _run_llava_symptom_analysis(image_path)

    # Plant identification - run async in sync context
    plant_name = _identify_plant(image_path)

    # Combine all
    all_results = _merge_results_multi(disease_results, nutrient_results, symptom_results)

    result = _map_to_knowledge_base(all_results)
    result["plant_name"] = plant_name
    return result

def _preprocess_disease_image(image_path: str) -> np.ndarray:
    """Preprocess image for disease TFLite model."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to convert logits to probabilities."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def _run_tflite_disease_inference(image_path: str, top_k: int = 5) -> Optional[List[Dict]]:
    """Run disease inference using TFLite model."""
    if _tflite_interpreter is None:
        return None

    try:
        input_details = _tflite_interpreter.get_input_details()
        output_details = _tflite_interpreter.get_output_details()

        input_data = _preprocess_disease_image(image_path)
        _tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        _tflite_interpreter.invoke()

        output_data = _tflite_interpreter.get_tensor(output_details[0]['index'])[0]

        # Apply softmax to convert logits to probabilities
        probabilities = _softmax(output_data)

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(probabilities[idx])
            label = DISEASE_CLASS_NAMES.get(idx, f"class_{idx}")
            results.append({"label": label, "score": score})

        logger.info(f"TFLite disease inference: {results[0]['label']} ({results[0]['score']:.2%})")
        return results

    except Exception as e:
        logger.error(f"TFLite disease inference failed: {e}")
        return None


def _run_huggingface_disease_inference(image_path: str, top_k: int = 5) -> Optional[List[Dict]]:
    """Run disease inference using HuggingFace pipeline."""
    pipeline = _load_huggingface_disease_model()
    if pipeline is None:
        return None

    try:
        predictions = pipeline(image_path, top_k=top_k)
        results = [{"label": p["label"], "score": p["score"]} for p in predictions]
        logger.info(f"HuggingFace disease inference: {results[0]['label']} ({results[0]['score']:.2%})")
        return results
    except Exception as e:
        logger.error(f"HuggingFace disease inference failed: {e}")
        return None


def _run_disease_model(image_path: str, top_k: int = 5) -> Dict:
    """
    Run disease detection with hybrid TFLite + HuggingFace approach.

    Strategy:
    1. Run TFLite (fast, local, 15 classes)
    2. If TFLite confidence < 90%, also run HuggingFace (slower, 38 classes)
    3. Pick the result with higher confidence

    This ensures we get the best of both models:
    - TFLite for fast inference on known diseases
    - HuggingFace for broader coverage (powdery mildew, apple diseases, etc.)
    """
    import os
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return {"diagnoses": [], "confidence": 0.0}

    # Run TFLite first (fast, local)
    tflite_predictions = _run_tflite_disease_inference(image_path, top_k)
    tflite_conf = tflite_predictions[0]["score"] if tflite_predictions else 0.0

    # If TFLite confidence is low (<90%), also try HuggingFace for broader coverage
    hf_predictions = None
    hf_conf = 0.0

    if tflite_conf < 0.90:
        logger.info(f"TFLite confidence ({tflite_conf:.1%}) < 90%, checking HuggingFace...")
        hf_predictions = _run_huggingface_disease_inference(image_path, top_k)
        hf_conf = hf_predictions[0]["score"] if hf_predictions else 0.0

    # Combine results from both models for best coverage
    # TFLite may not have all disease classes, HuggingFace has broader coverage
    all_predictions = []

    if tflite_predictions:
        for p in tflite_predictions:
            p["model_source"] = "tflite"
            # When in hybrid mode, TFLite was uncertain (<90%) and only has 15 classes.
            # Apply a small penalty so HuggingFace (38 classes) is preferred when scores are close.
            if hf_predictions:
                p["score"] = p["score"] * 0.95
        all_predictions.extend(tflite_predictions)

    if hf_predictions:
        for p in hf_predictions:
            p["model_source"] = "huggingface"
        all_predictions.extend(hf_predictions)

    if not all_predictions:
        logger.warning("All disease detection methods failed")
        return {"diagnoses": [], "confidence": 0.0}

    # Sort by confidence and take top results
    all_predictions.sort(key=lambda x: x["score"], reverse=True)
    predictions = all_predictions[:top_k]
    source = "hybrid" if (tflite_predictions and hf_predictions) else predictions[0].get("model_source", "unknown")
    logger.info(f"Combined predictions: TFLite={len(tflite_predictions or [])} + HuggingFace={len(hf_predictions or [])}")

    diagnoses = []
    for pred in predictions:
        raw_label = pred.get('label', '')
        # First try normalized mapping
        diagnosis_info = get_diagnosis_info(raw_label)

        # Fallback: extract disease substring with multiple strategies
        if diagnosis_info is None:
            try:
                lower = raw_label.lower()
                disease_part = None

                # Strategy 1: HuggingFace format "X with Y"
                if ' with ' in lower:
                    disease_part = raw_label.split(' with ', 1)[1]
                elif ' of ' in lower:
                    disease_part = raw_label.split(' of ', 1)[1]

                # Strategy 2: TFLite format "Tomato___Early_blight" or "Tomato_Early_blight"
                elif '___' in raw_label:
                    disease_part = raw_label.split('___')[-1].replace('_', ' ')
                elif '__' in raw_label:
                    disease_part = raw_label.split('__')[-1].replace('_', ' ')
                elif '_' in raw_label and any(crop in lower for crop in ['tomato', 'potato', 'pepper', 'grape', 'cherry']):
                    # Handle "Tomato_Late_blight" format
                    parts = raw_label.split('_')
                    # Skip the crop name (first part)
                    disease_part = '_'.join(parts[1:]).replace('_', ' ')

                # Strategy 3: Healthy detection
                if 'healthy' in lower:
                    disease_part = 'healthy'

                if disease_part:
                    # Try the extracted part
                    diagnosis_info = get_diagnosis_info(disease_part)

                    # If still not found, try with underscores
                    if diagnosis_info is None:
                        disease_part_underscore = disease_part.lower().replace(' ', '_')
                        diagnosis_info = get_diagnosis_info(disease_part_underscore)

                    if diagnosis_info:
                        logger.debug(f"Mapped label '{raw_label}' -> '{disease_part}'")
            except Exception:
                diagnosis_info = None

        if diagnosis_info:
            diagnoses.append({
                "source": source,
                "label": normalize_label(raw_label),
                "confidence": pred['score'],
                "category": diagnosis_info.category,
                "subcategory": diagnosis_info.subcategory,
                "info": diagnosis_info
            })
        else:
            logger.warning(f"No mapping found for label: {raw_label}")

    if not diagnoses:
        logger.warning("All disease predictions failed to map to database")
    else:
        logger.info(f"Successfully mapped {len(diagnoses)} diagnoses")

    max_confidence = max([p['score'] for p in predictions]) if predictions else 0.0

    return {
        "diagnoses": diagnoses,
        "confidence": max_confidence,
        "source": source,
        "model_format": _disease_model_format
    }

def _run_llava_symptom_analysis(image_path: str) -> Dict:
    """LLaVA for nutrient deficiencies + symptom description"""
    prompt = """Analyze this plant leaf. Identify:
1. Main symptom: yellowing, spots, wilting, etc.
2. Pattern: old/new leaves, margins, between veins
3. Likely cause: N/P/K/Fe/Mg/Ca deficiency OR fungal/bacterial

Format: SYMPTOM: ... | PATTERN: ... | DIAGNOSIS: Iron deficiency"""

    try:
        # Read and encode image as base64 for LLaVA vision model
        import base64
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode()

        # Use chat_completion for LLaVA vision-language model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ]

        result = llava_client.chat_completion(messages=messages, max_tokens=200)
        response_text = result.choices[0].message.content if result.choices else ""

        parsed = _parse_llava_response(response_text)
        return {
            "diagnoses": [parsed] if parsed else [],
            "confidence": 0.75
        }
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return {"diagnoses": [], "confidence": 0.0}
    except Exception as e:
        # Enhanced error logging
        error_type = type(e).__name__
        error_msg = str(e)

        logger.warning(
            f"LLaVA analysis failed ({error_type}): {error_msg}. "
            "Continuing with MobileNet-only diagnosis."
        )

        # Log specific API errors for debugging
        if "410" in error_msg or "Gone" in error_msg:
            logger.error("Hugging Face API endpoint deprecated - check configuration")
        elif "401" in error_msg or "403" in error_msg:
            logger.error("Authentication error - check HF_TOKEN configuration")
        elif "503" in error_msg or "timeout" in error_msg.lower():
            logger.warning("Hugging Face API temporarily unavailable")

        return {"diagnoses": [], "confidence": 0.0}
    
def _merge_results_multi(*result_dicts: Dict) -> List[Dict]:
    """Merge multiple result dicts (disease, coleaf, llava) then reuse _merge_results logic."""
    combined = {"diagnoses": [], "confidence": 0.0}
    for rd in result_dicts:
        combined["diagnoses"].extend(rd.get("diagnoses", []))
    return _merge_results(combined, {"diagnoses": [], "confidence": 0.0})

def _parse_llava_response(text: str) -> Dict:
    """Extract structured diagnosis from LLaVA text"""
    # Simple regex/keyword matching - improve as needed
    text_lower = text.lower()
    
    # Nutrient deficiency patterns
    nutrient_map = {
        "nitrogen": "N", "n deficiency": "N",
        "phosphorus": "P", "p deficiency": "P", "purple": "P",
        "potassium": "K", "k deficiency": "K", "brown edges": "K",
        "iron": "Fe", "fe deficiency": "Fe", "yellow new leaves": "Fe",
        "magnesium": "Mg", "mg deficiency": "Mg", "yellow old veins green": "Mg",
        "calcium": "Ca", "ca deficiency": "Ca", "tip burn": "Ca"
    }
    
    for symptom, nutrient in nutrient_map.items():
        if symptom in text_lower:
            return {
                "source": "llava",
                "label": f"{nutrient}_deficiency",
                "confidence": 0.8,
                "category": "nutrient_deficiency",
                "subcategory": nutrient
            }
    
    # Fungal/bacterial fallback
    if any(word in text_lower for word in ["powdery", "white spots", "mildew"]):
        return {"source": "llava", "label": "powdery_mildew", "confidence": 0.75, "category": "fungal"}
    if "yellow halo" in text_lower or "bacterial" in text_lower:
        return {"source": "llava", "label": "bacterial_spot", "confidence": 0.75, "category": "bacterial"}
    
    return {"source": "llava", "label": "general_plant_stress", "confidence": 0.5, "category": "environmental"}

def _merge_results(disease_results: Dict, symptom_results: Dict) -> List[Dict]:
    """Intelligent merging: Disease > High-conf LLaVA > Rules"""
    all_diagnoses = disease_results.get("diagnoses", []) + symptom_results.get("diagnoses", [])

    if not all_diagnoses:
        logger.warning("No diagnoses from either model, returning empty result")
        return []

    # Prioritize: High-conf disease (TFLite/MobileNet) > LLaVA nutrients > low-conf
    sorted_diagnoses = sorted(all_diagnoses, key=lambda x: (
        1 if x.get("source") in ["mobilenet", "tflite", "huggingface"] and x.get("confidence", 0) > 0.7 else
        2 if x.get("source") == "llava" and x.get("confidence", 0) > 0.75 else
        3,
        -x.get("confidence", 0)  # Secondary sort by confidence (descending)
    ))

    return sorted_diagnoses[:3]  # Top 3 diagnoses

def _map_to_knowledge_base(hybrid_results: List[Dict]) -> Dict:
    """Map hybrid results to your rich DISEASE_MAPPINGS"""
    if not hybrid_results:
        logger.info("No diagnosis results to map, returning healthy plant response")
        return {
            "success": True,
            "plant_health_score": 85,
            "primary_diagnosis": None,
            "all_diagnoses": [],
            "diagnoses": [],
            "recommendations": [
                "Your plant appears healthy! Continue regular care.",
                "Maintain consistent watering schedule",
                "Ensure adequate light exposure",
                "Monitor for any changes in appearance"
            ],
            "ai_sources_used": [],
            "plant_name": None
        }

    diagnoses = []
    for result in hybrid_results:
        label = result.get("label", "")
        info = get_diagnosis_info(label)
        if info:
            diagnoses.append({
                "source": result.get("source", "unknown"),
                "label": label,
                "confidence": result.get("confidence", 0),
                "category": info.category,
                "subcategory": info.subcategory,
                "name": info.name,
                "symptoms": info.symptoms,
                "causes": info.causes,
                "treatment": info.treatment,
                "severity": _assess_severity(result.get("confidence", 0), info)
            })
        else:
            logger.warning(f"No mapping found for label: {label}")

    # Fallback if no valid diagnoses after mapping
    if not diagnoses:
        logger.warning("All diagnoses failed to map to knowledge base")
        return {
            "success": False,
            "plant_health_score": 50,
            "primary_diagnosis": None,
            "all_diagnoses": [],
            "diagnoses": [],
            "recommendations": [
                "Unable to identify specific issue from the image",
                "Try taking a clearer photo in good lighting",
                "Focus on affected leaves or problem areas",
                "Consult a local plant expert if issues persist"
            ],
            "ai_sources_used": [d.get("source", "unknown") for d in hybrid_results],
            "plant_name": None
        }

    # Your existing logic for health score + recommendations
    health_score = _calculate_health_score(diagnoses)
    recommendations = _generate_recommendations(diagnoses)

    # Get model version IDs for tracking
    model_versions = {}
    coleaf_version = get_coleaf_version()
    if coleaf_version:
        model_versions["coleaf"] = coleaf_version

    return {
        "success": True,
        "plant_health_score": health_score,
        "primary_diagnosis": diagnoses[0] if diagnoses else None,
        "all_diagnoses": diagnoses,
        "diagnoses": diagnoses,  # For frontend compatibility
        "recommendations": recommendations,
        "ai_sources_used": list(set([d.get("source", "unknown") for d in diagnoses])),
        "plant_name": None,  # Will be set by diagnose_image after identification
        "model_versions": model_versions,  # For continuous learning tracking
    }

def _assess_severity(confidence: float, info=None) -> str:
    """Assess severity based on confidence and diagnosis info"""
    # Could use info.severity_indicators in future for more nuanced assessment
    if confidence >= 0.8:
        return "severe"
    elif confidence >= 0.5:
        return "moderate"
    else:
        return "mild"

def _calculate_health_score(diagnoses: List[Dict]) -> int:
    """Calculate overall plant health score (0-100)"""
    if not diagnoses:
        return 85  # Default healthy score if no issues detected

    # Base score starts at 100
    health_score = 100

    for diagnosis in diagnoses:
        confidence = diagnosis.get("confidence", 0)
        category = diagnosis.get("category", "")
        severity = diagnosis.get("severity", "mild")

        # Deduct points based on severity and confidence
        if severity == "severe":
            health_score -= int(confidence * 40)  # Up to 40 points
        elif severity == "moderate":
            health_score -= int(confidence * 25)  # Up to 25 points
        else:  # mild
            health_score -= int(confidence * 15)  # Up to 15 points

        # Additional deductions for specific categories
        if category == "bacterial" or category == "fungal":
            health_score -= 10  # Diseases are more serious
        elif category == "nutrient_deficiency":
            health_score -= 5  # Deficiencies are less critical

    # Clamp between 0 and 100
    return max(0, min(100, health_score))

def _generate_recommendations(diagnoses: List[Dict]) -> List[str]:
    """Generate actionable recommendations based on diagnoses"""
    if not diagnoses:
        return [
            "Your plant appears healthy! Continue regular care.",
            "Maintain consistent watering schedule",
            "Ensure adequate light exposure",
            "Monitor for any changes in appearance"
        ]

    recommendations = []
    primary = diagnoses[0] if diagnoses else None

    if not primary:
        return recommendations

    category = primary.get("category", "")
    severity = primary.get("severity", "mild")
    treatment = primary.get("treatment", [])

    # Add urgency message based on severity
    if severity == "severe":
        recommendations.append("⚠️ URGENT: Immediate action required to save your plant")
    elif severity == "moderate":
        recommendations.append("⚡ Act soon: Address this issue within the next few days")
    else:
        recommendations.append("✓ Monitor closely and apply treatments as needed")

    # Add top 3 treatments from diagnosis info
    if treatment:
        recommendations.extend(treatment[:3])

    # Add category-specific recommendations
    if category == "nutrient_deficiency":
        recommendations.append("💡 Tip: Get a soil test for precise nutrient levels")
        recommendations.append("Consider using a balanced fertilizer")
    elif category == "fungal":
        recommendations.append("🍃 Improve air circulation around the plant")
        recommendations.append("Avoid overhead watering")
    elif category == "bacterial":
        recommendations.append("🧼 Sanitize tools before and after use")
        recommendations.append("Remove infected plant material immediately")
    elif category == "environmental":
        recommendations.append("🌡️ Check environmental conditions (light, temp, humidity)")
        recommendations.append("Ensure proper drainage and avoid overwatering")

    # Add general monitoring advice
    recommendations.append("📸 Take photos to track progress over time")

    return recommendations[:8]  # Limit to 8 recommendations