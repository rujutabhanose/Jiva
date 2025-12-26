from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
import json
from app.api.deps import get_db, get_current_user
from app.models import user as user_model
from app.models.scan import Scan
from app.services.image_utils import save_image

router = APIRouter()


class CreateScanRequest(BaseModel):
    device_id: str
    mode: str  # 'diagnosis' | 'identification'
    image_url: Optional[str] = None  # Base64 or URL
    condition_name: str
    confidence: float
    diagnosis_data: Optional[Dict[str, Any]] = None
    symptoms: Optional[List[str]] = None
    causes: Optional[List[str]] = None
    treatment: Optional[List[str]] = None
    notes: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    health_score: Optional[float] = None


class UpdateScanNotesRequest(BaseModel):
    notes: str


class ScanResponse(BaseModel):
    id: int
    user_id: int
    device_id: str
    mode: str
    image_url: Optional[str]
    condition_name: str
    confidence: float
    diagnosis_data: Optional[Dict[str, Any]]
    symptoms: Optional[List[str]]
    causes: Optional[List[str]]
    treatment: Optional[List[str]]
    notes: Optional[str]
    category: Optional[str]
    severity: Optional[str]
    health_score: Optional[float]
    created_at: str
    updated_at: str


@router.post("/", response_model=ScanResponse)
async def create_scan(
    device_id: str = Form(...),
    mode: str = Form(...),
    condition_name: str = Form(...),
    confidence: float = Form(...),
    diagnosis_data: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
    causes: Optional[str] = Form(None),
    treatment: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    health_score: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a new scan to the database with optional image upload (requires authentication)"""

    # Handle image upload if provided
    image_url = None
    if image:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")
        image_path = save_image(image)
        image_url = f"/uploads/{Path(image_path).name}"

    # Parse JSON fields from form data
    diagnosis_data_dict = json.loads(diagnosis_data) if diagnosis_data else None
    symptoms_list = json.loads(symptoms) if symptoms else None
    causes_list = json.loads(causes) if causes else None
    treatment_list = json.loads(treatment) if treatment else None

    # Create scan using authenticated user's ID
    scan = Scan(
        user_id=current_user.id,
        device_id=device_id,
        mode=mode,
        image_url=image_url,
        condition_name=condition_name,
        confidence=confidence,
        diagnosis_data=diagnosis_data_dict,
        symptoms=symptoms_list,
        causes=causes_list,
        treatment=treatment_list,
        notes=notes,
        category=category,
        severity=severity,
        health_score=health_score
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "device_id": scan.device_id,
        "mode": scan.mode,
        "image_url": scan.image_url,
        "condition_name": scan.condition_name,
        "confidence": scan.confidence,
        "diagnosis_data": scan.diagnosis_data,
        "symptoms": scan.symptoms,
        "causes": scan.causes,
        "treatment": scan.treatment,
        "notes": scan.notes,
        "category": scan.category,
        "severity": scan.severity,
        "health_score": scan.health_score,
        "created_at": scan.created_at.isoformat(),
        "updated_at": scan.updated_at.isoformat()
    }


@router.get("", include_in_schema=False)
async def get_scans_noslash(
    mode: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept GET requests to /api/v1/scans (no trailing slash) and forward to canonical handler"""
    return await get_scans(mode=mode, limit=limit, offset=offset, current_user=current_user, db=db)


@router.get("/", response_model=List[ScanResponse])
async def get_scans(
    mode: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get scan history for the authenticated user"""
    # Build query - ALWAYS filter by authenticated user's ID
    query = db.query(Scan).filter(Scan.user_id == current_user.id)

    if mode:
        query = query.filter(Scan.mode == mode)

    # Order by most recent first
    query = query.order_by(Scan.created_at.desc())

    # Apply pagination
    scans = query.offset(offset).limit(limit).all()

    return [
        {
            "id": scan.id,
            "user_id": scan.user_id,
            "device_id": scan.device_id,
            "mode": scan.mode,
            "image_url": scan.image_url,
            "condition_name": scan.condition_name,
            "confidence": scan.confidence,
            "diagnosis_data": scan.diagnosis_data,
            "symptoms": scan.symptoms,
            "causes": scan.causes,
            "treatment": scan.treatment,
            "notes": scan.notes,
            "category": scan.category,
            "severity": scan.severity,
            "health_score": scan.health_score,
            "created_at": scan.created_at.isoformat(),
            "updated_at": scan.updated_at.isoformat()
        }
        for scan in scans
    ]


@router.get("/{scan_id}", response_model=ScanResponse)
async def get_scan(
    scan_id: int,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific scan by ID (must belong to authenticated user)"""
    scan = db.query(Scan).filter(
        Scan.id == scan_id,
        Scan.user_id == current_user.id
    ).first()

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "device_id": scan.device_id,
        "mode": scan.mode,
        "image_url": scan.image_url,
        "condition_name": scan.condition_name,
        "confidence": scan.confidence,
        "diagnosis_data": scan.diagnosis_data,
        "symptoms": scan.symptoms,
        "causes": scan.causes,
        "treatment": scan.treatment,
        "notes": scan.notes,
        "category": scan.category,
        "severity": scan.severity,
        "health_score": scan.health_score,
        "created_at": scan.created_at.isoformat(),
        "updated_at": scan.updated_at.isoformat()
    }


@router.put("/{scan_id}/notes")
async def update_scan_notes(
    scan_id: int,
    request: UpdateScanNotesRequest,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update notes for a specific scan (must belong to authenticated user)"""
    scan = db.query(Scan).filter(
        Scan.id == scan_id,
        Scan.user_id == current_user.id
    ).first()

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    scan.notes = request.notes
    db.commit()
    db.refresh(scan)

    return {
        "id": scan.id,
        "notes": scan.notes,
        "updated_at": scan.updated_at.isoformat()
    }


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: int,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a specific scan (must belong to authenticated user)"""
    scan = db.query(Scan).filter(
        Scan.id == scan_id,
        Scan.user_id == current_user.id
    ).first()

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Delete associated image file if it exists
    if scan.image_url:
        try:
            # Extract filename from URL path (e.g., "/uploads/abc123.jpg" -> "abc123.jpg")
            image_filename = Path(scan.image_url).name
            image_path = Path("uploads") / image_filename

            if image_path.exists():
                image_path.unlink()  # Delete the file
        except Exception as e:
            # Log error but don't fail the deletion
            print(f"Warning: Could not delete image file {scan.image_url}: {e}")

    db.delete(scan)
    db.commit()

    return {"message": "Scan deleted successfully"}


@router.get("/stats/summary")
async def get_scan_stats(
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get scan statistics for the authenticated user"""
    query = db.query(Scan).filter(Scan.user_id == current_user.id)

    total_scans = query.count()
    diagnosis_scans = query.filter(Scan.mode == "diagnosis").count()
    identification_scans = query.filter(Scan.mode == "identification").count()

    return {
        "total_scans": total_scans,
        "diagnosis_scans": diagnosis_scans,
        "identification_scans": identification_scans,
        "scans_by_mode": {
            "diagnosis": diagnosis_scans,
            "identification": identification_scans
        }
    }
