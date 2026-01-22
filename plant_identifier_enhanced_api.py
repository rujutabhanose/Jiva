#!/usr/bin/env python3
"""
Enhanced FastAPI router for plant identification
Integrates ensemble model + post-processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil

from plant_identifier_ensemble import EnsemblePlantIdentifier
from plant_id_postprocessor import PlantIdentificationPostProcessor

# Initialize models (lazy loading)
identifier = None
postprocessor = None

def get_identifier():
    global identifier
    if identifier is None:
        identifier = EnsemblePlantIdentifier()
    return identifier

def get_postprocessor():
    global postprocessor
    if postprocessor is None:
        postprocessor = PlantIdentificationPostProcessor()
    return postprocessor

# Create router
plant_id_router = APIRouter(prefix="/identify-enhanced", tags=["Plant Identification"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@plant_id_router.post("/")
async def identify_enhanced(image: UploadFile = File(...)):
    """
    Enhanced plant identification endpoint

    Request:
        - image: Image file (multipart/form-data)

    Response:
        {
            "status": "success",
            "data": {
                "primary": {
                    "scientific_name": "Epipremnum aureum",
                    "common_name": "Money Plant",
                    "confidence": 98.7
                },
                "alternatives": [...],
                "confidence_level": "very_high",
                "recommendation": "Care instructions...",
                "similar_species": [...]
            }
        }
    """

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail={
            "error": "Invalid file type. Please upload an image.",
            "code": "INVALID_FILE_TYPE"
        })

    # Save temporarily
    temp_path = UPLOAD_DIR / f"temp_{image.filename}"

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Identify with ensemble
        raw_result = get_identifier().identify_single(str(temp_path), top_k=5, debug=False)

        # Post-process
        final_result = get_postprocessor().process_result(raw_result)

        return {
            "status": "success",
            "data": final_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "error": str(e),
            "code": "IDENTIFICATION_FAILED"
        })

    finally:
        # Clean up
        if temp_path.exists():
            os.remove(temp_path)


@plant_id_router.post("/batch")
async def identify_batch(images: list[UploadFile] = File(...)):
    """
    Identify multiple plants at once

    Request:
        - images: List of image files (multipart/form-data)

    Response:
        List of identification results
    """

    results = []
    temp_paths = []

    try:
        for image in images:
            if not image.content_type or not image.content_type.startswith("image/"):
                continue

            temp_path = UPLOAD_DIR / f"temp_{image.filename}"
            temp_paths.append(temp_path)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            raw_result = get_identifier().identify_single(str(temp_path), top_k=3, debug=False)
            final_result = get_postprocessor().process_result(raw_result)

            results.append({
                "filename": image.filename,
                "result": final_result
            })

        return {
            "status": "success",
            "count": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "error": str(e)
        })

    finally:
        # Clean up all temp files
        for temp_path in temp_paths:
            if temp_path.exists():
                os.remove(temp_path)


@plant_id_router.get("/health")
async def health():
    """Health check for enhanced plant ID"""
    try:
        models_loaded = len(get_identifier().models) if identifier else 0
        threshold = get_postprocessor().confidence_threshold if postprocessor else 35.0
    except Exception:
        models_loaded = 0
        threshold = 35.0

    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "confidence_threshold": threshold
    }
