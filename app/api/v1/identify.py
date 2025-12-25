from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel
from typing import List
import logging
import os
from sqlalchemy.orm import Session

from app.services.plant_identifier import identify_plant
from app.services.image_utils import save_image
from app.api.deps import get_db
from app.models.user import User

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
    device_id: str = Form(...),
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    user = db.query(User).filter(User.device_id == device_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ✅ Plant identification is FREE - no scan limit enforcement
    # Only diagnosis requires paid scans

    image_path = None
    try:
        image_path = save_image(file)
        result_dict = identify_plant(image_path)

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
            "free_scans_left": user.free_scans_left,
            "is_premium": user.is_premium,
        }

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)