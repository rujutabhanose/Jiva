from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import tempfile
from sqlalchemy.orm import Session

from app.services.hybrid_plant_identifier import identify_plant_hybrid
from app.services.quality_gate import image_quality_check
from app.api.deps import get_db, get_current_user
from app.models import user as user_model

logger = logging.getLogger(__name__)
router = APIRouter()


class PlantIdentification(BaseModel):
    plant_name: str
    scientific_name: Optional[str] = None
    family: Optional[str] = None
    confidence: float
    confidence_percent: float


class IdentifyResponse(BaseModel):
    success: bool
    results: List[PlantIdentification]
    free_scans_left: int
    is_premium: bool
    reason: Optional[str] = None


@router.post("/", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(...),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Plant identification endpoint (requires authentication, FREE for all users)

    Uses hybrid approach: tries both local and cloud models, returns highest confidence.
    """

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

        # Quality gate check
        if not image_quality_check(temp_path):
            return {
                "success": False,
                "results": [],
                "free_scans_left": current_user.free_scans_left,
                "is_premium": current_user.is_premium,
                "reason": "Image too blurry or unclear. Please take a closer, clearer photo of the leaf."
            }

        # Use hybrid identification (tries both local and cloud models)
        try:
            result_dict = await identify_plant_hybrid(temp_path, region="India")
        except RuntimeError as e:
            logger.error(f"Plant identification failed: {e}")
            raise HTTPException(
                status_code=503,
                detail="Plant identification service temporarily unavailable. Please try again later."
            )

        # Convert the result format from {primary, alternatives} to a list
        # Primary result should be first in the list
        primary = result_dict["primary"]
        alternatives = result_dict.get("alternatives", [])

        # Format primary result with USDA enrichment data
        results = [{
            "plant_name": primary.get("commonName") or primary.get("scientificName"),
            "scientific_name": primary.get("scientificName"),
            "family": primary.get("family"),
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