from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from pathlib import Path
from app.services import diagnosis_engine
from app.services.scan_limits import enforce_and_consume_scan
from app.services.image_utils import save_image  # ← NEW
from app.api.deps import get_db
from app.models.user import User

router = APIRouter()

@router.post("", include_in_schema=False)
async def diagnose_plant_noslash(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Accepts POST requests to /api/v1/diagnose (no trailing slash)"""
    return await diagnose_plant(file=file, device_id=device_id, db=db)

@router.post("/")
async def diagnose_plant(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Hybrid diagnosis endpoint - MobileNetV2 + LLaVA + Rich remedies"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    # 🔍 BUSINESS LOGIC (UNCHANGED)
    user = db.query(User).filter(User.device_id == device_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    enforce_and_consume_scan(user, db)  # Scan limits preserved ✅

    # 🆕 NEW: Use your image_utils (persistent + temp handling)
    image_path = save_image(file)  # Saves to uploads/abc123.jpg
    
    try:
        # 🧠 HYBRID ENGINE (MobileNetV2 + LLaVA)
        result = diagnosis_engine.diagnose_image(image_path)
        
        # 📊 USER STATS (UNCHANGED)
        result["free_scans_left"] = user.free_scans_left
        result["is_premium"] = user.is_premium
        result["image_url"] = f"/uploads/{Path(image_path).name}"  # History thumbnail ✅
        
        return result
        
    finally:
        # Temp files auto-cleaned by image_utils
        pass