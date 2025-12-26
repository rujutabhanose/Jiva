from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from pathlib import Path
import tempfile
import os
from app.services import diagnosis_engine
from app.services.scan_limits import enforce_and_consume_scan
from app.api.deps import get_db, get_current_user
from app.models import user as user_model

router = APIRouter()

@router.post("", include_in_schema=False)
async def diagnose_plant_noslash(
    file: UploadFile = File(...),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Accepts POST requests to /api/v1/diagnose (no trailing slash)"""
    return await diagnose_plant(file=file, current_user=current_user, db=db)

@router.post("/")
async def diagnose_plant(
    file: UploadFile = File(...),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Hybrid diagnosis endpoint - MobileNetV2 + LLaVA + Rich remedies (requires authentication)"""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    # Enforce scan limits for authenticated user
    enforce_and_consume_scan(current_user, db)

    # Create temporary file for image processing (NOT saved permanently)
    image_content = await file.read()
    fd, temp_path = tempfile.mkstemp(suffix='.jpg')

    try:
        # Write image to temp file
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(image_content)

        # 🧠 HYBRID ENGINE (MobileNetV2 + LLaVA)
        result = diagnosis_engine.diagnose_image(temp_path)

        # 📊 USER STATS
        result["free_scans_left"] = current_user.free_scans_left
        result["is_premium"] = current_user.is_premium
        # NOTE: No image_url returned - image will be saved when user clicks "Save"

        return result

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)