"""
Hybrid Plant Diagnosis Engine v2.0
Combines MobileNetV2 (diseases) + LLaVA (symptoms/nutrients) + Rule heuristics
"""

from transformers import pipeline
from huggingface_hub import InferenceClient
from typing import List, Dict, Optional
import logging
import re
from .disease_mapping import (
    get_diagnosis_info, normalize_label, is_nutrient_deficiency,
    is_fungal_disease, is_bacterial_disease, get_nutrient_type
)

logger = logging.getLogger(__name__)

# Load models
disease_model = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

# LLaVA client for symptom analysis (nutrient deficiencies + broad coverage)
# Using new Hugging Face router endpoint (api-inference.huggingface.co is deprecated)
llava_client = InferenceClient(
    model="YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
)

def diagnose_image(image_path: str, top_k: int = 5) -> Dict:
    """Hybrid diagnosis: MobileNetV2 + LLaVA + Rules"""

    # Parallel processing
    disease_results = _run_disease_model(image_path, top_k)
    # LLaVA temporarily disabled due to deprecated Hugging Face endpoint
    # TODO: Re-enable when model is migrated to router.huggingface.co
    symptom_results = {"diagnoses": [], "confidence": 0.0}
    # symptom_results = _run_llava_symptom_analysis(image_path)
    
    # Combine + prioritize
    hybrid_diagnosis = _merge_results(disease_results, symptom_results)
    
    # Map to your rich DISEASE_MAPPINGS + remedies
    final_diagnosis = _map_to_knowledge_base(hybrid_diagnosis)
    
    return final_diagnosis

def _run_disease_model(image_path: str, top_k: int = 5) -> Dict:
    """Your existing MobileNetV2 logic (diseases)"""
    try:
        # Check if image file exists
        import os
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"diagnoses": [], "confidence": 0.0}

        logger.info(f"Running MobileNetV2 on image: {image_path}")
        predictions = disease_model(image_path, top_k=top_k)

        if not predictions:
            logger.warning("MobileNetV2 returned no predictions")
            return {"diagnoses": [], "confidence": 0.0}

        logger.info(f"MobileNetV2 returned {len(predictions)} predictions")
        logger.debug(f"Raw predictions: {predictions}")

        diagnoses = []
        for pred in predictions:
            diagnosis_info = get_diagnosis_info(pred['label'])
            if diagnosis_info:
                diagnoses.append({
                    "source": "mobilenet",
                    "label": pred['label'],
                    "confidence": pred['score'],
                    "category": diagnosis_info.category,
                    "subcategory": diagnosis_info.subcategory,
                    "info": diagnosis_info
                })
            else:
                logger.warning(f"No mapping found for MobileNet label: {pred['label']}")

        if not diagnoses:
            logger.warning("All MobileNet predictions failed to map to disease database")
        else:
            logger.info(f"Successfully mapped {len(diagnoses)} diagnoses")

        # Calculate max confidence from predictions
        max_confidence = max([p['score'] for p in predictions]) if predictions else 0.0

        return {"diagnoses": diagnoses, "confidence": max_confidence}

    except Exception as e:
        logger.error(f"Error in disease model: {e}", exc_info=True)
        return {"diagnoses": [], "confidence": 0.0}

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

    # Prioritize: High-conf disease > LLaVA nutrients > low-conf disease
    sorted_diagnoses = sorted(all_diagnoses, key=lambda x: (
        1 if x.get("source") == "mobilenet" and x.get("confidence", 0) > 0.7 else
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
            "ai_sources_used": []
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
            "ai_sources_used": [d.get("source", "unknown") for d in hybrid_results]
        }

    # Your existing logic for health score + recommendations
    health_score = _calculate_health_score(diagnoses)
    recommendations = _generate_recommendations(diagnoses)

    return {
        "success": True,
        "plant_health_score": health_score,
        "primary_diagnosis": diagnoses[0] if diagnoses else None,
        "all_diagnoses": diagnoses,
        "diagnoses": diagnoses,  # For frontend compatibility
        "recommendations": recommendations,
        "ai_sources_used": list(set([d.get("source", "unknown") for d in diagnoses]))
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