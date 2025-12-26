from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import logging
import os
import tempfile
from sqlalchemy.orm import Session

from app.services.plant_identifier import identify_plant
from app.api.deps import get_db, get_current_user
from app.models import user as user_model

logger = logging.getLogger(__name__)
router = APIRouter()


class PlantIdentification(BaseModel):
    plant_name: str
    confidence: float
    confidence_percent: float


class IdentifyResponse(BaseModel):
    success: bool
    results: List[PlantIdentification]
    free_scans_left: int
    is_premium: bool


@router.post("/", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Plant identification endpoint (requires authentication, FREE for all users)"""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    # ✅ Plant identification is FREE - no scan limit enforcement
    # Only diagnosis requires paid scans

    # Create temporary file for image processing (NOT saved permanently)
    image_content = await file.read()
    fd, temp_path = tempfile.mkstemp(suffix='.jpg')

    try:
        # Write image to temp file
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(image_content)

        result_dict = identify_plant(temp_path)

        # Convert the result format from {primary, alternatives} to a list
        # Primary result should be first in the list
        primary = result_dict["primary"]
        alternatives = result_dict.get("alternatives", [])

        # Format primary result
        results = [{
            "plant_name": primary.get("commonName") or primary.get("scientificName"),
            "confidence": primary["confidence"] / 100 if primary["confidence"] > 1 else primary["confidence"],
            "confidence_percent": primary["confidence"] if primary["confidence"] > 1 else primary["confidence"] * 100
        }]

        # Add alternatives
        results.extend(alternatives)

        return {
            "success": True,
            "results": results,
            "free_scans_left": current_user.free_scans_left,
            "is_premium": current_user.is_premium,
        }

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)