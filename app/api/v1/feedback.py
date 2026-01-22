from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import logging

from app.api.deps import get_db, get_current_user
from app.models import user as user_model
from app.models.diagnosis_feedback import DiagnosisFeedback
from app.models.scan import Scan
from app.services.training_data_service import TrainingDataService

router = APIRouter()
logger = logging.getLogger(__name__)


class DiagnosisFeedbackRequest(BaseModel):
    scan_id: Optional[int] = None
    original_condition: str
    is_correct: bool
    corrected_condition: Optional[str] = None
    # New fields for continuous learning
    model_type: Optional[str] = None  # "coleaf", "disease", "plant_id"
    confidence: Optional[float] = None


class DiagnosisFeedbackResponse(BaseModel):
    id: int
    message: str
    training_sample_saved: bool = False


async def save_training_sample_background(
    user_id: int,
    scan_id: Optional[int],
    confirmed_label: str,
    model_type: str,
    image_url: str,
    original_prediction: Optional[str],
    original_confidence: Optional[float],
    db: Session,
):
    """Background task to upload image and save training sample."""
    try:
        training_service = TrainingDataService(db)
        await training_service.save_confirmed_sample(
            user_id=user_id,
            scan_id=scan_id,
            confirmed_label=confirmed_label,
            model_type=model_type,
            image_source=image_url,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
        )
    except Exception as e:
        logger.error(f"Background training sample save failed: {e}")


@router.post("/diagnosis", response_model=DiagnosisFeedbackResponse)
async def submit_diagnosis_feedback(
    request: DiagnosisFeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit feedback on diagnosis accuracy.

    When user confirms diagnosis is correct (is_correct=True),
    the image and label are saved for model retraining.
    """
    training_sample_saved = False
    image_url = None

    # Get scan details if scan_id provided
    scan = None
    if request.scan_id:
        scan = db.query(Scan).filter(Scan.id == request.scan_id).first()
        if scan:
            image_url = scan.image_url

    # Determine model type from condition if not provided
    model_type = request.model_type
    if not model_type:
        # Infer model type from condition name
        condition_lower = request.original_condition.lower()
        if any(x in condition_lower for x in ["deficiency", "nitrogen", "phosphorus", "potassium", "iron", "magnesium", "calcium"]):
            model_type = "coleaf"
        elif any(x in condition_lower for x in ["blight", "mildew", "spot", "rot", "fungal", "bacterial", "virus"]):
            model_type = "disease"
        else:
            model_type = "disease"  # Default to disease model

    # Create feedback record
    feedback = DiagnosisFeedback(
        user_id=current_user.id,
        scan_id=request.scan_id,
        original_condition=request.original_condition,
        original_confidence=request.confidence,
        is_correct=request.is_correct,
        corrected_condition=request.corrected_condition if not request.is_correct else None,
        model_type=model_type,
        image_url=image_url,
        used_for_training=False,
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    # If diagnosis confirmed correct and we have an image, save for training
    if request.is_correct and image_url:
        # Determine the confirmed label
        confirmed_label = request.original_condition

        # Save training sample in background to not block response
        background_tasks.add_task(
            save_training_sample_background,
            user_id=current_user.id,
            scan_id=request.scan_id,
            confirmed_label=confirmed_label,
            model_type=model_type,
            image_url=image_url,
            original_prediction=request.original_condition,
            original_confidence=request.confidence,
            db=db,
        )
        training_sample_saved = True

        logger.info(
            f"Queued training sample: user={current_user.id}, "
            f"label={confirmed_label}, model={model_type}"
        )

    # If user provided correction, we could also save that for review
    # (but not automatically for training - needs manual review)
    if not request.is_correct and request.corrected_condition:
        logger.info(
            f"Received correction: {request.original_condition} -> "
            f"{request.corrected_condition} (user={current_user.id})"
        )

    message = (
        "Thank you for confirming! Your feedback helps our AI learn."
        if request.is_correct
        else "Thank you for your feedback! We'll use this to improve."
    )

    return {
        "id": feedback.id,
        "message": message,
        "training_sample_saved": training_sample_saved,
    }


@router.get("/stats")
async def get_feedback_stats(
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feedback statistics (for admin dashboard)."""
    total_feedback = db.query(DiagnosisFeedback).count()
    correct_count = db.query(DiagnosisFeedback).filter(
        DiagnosisFeedback.is_correct == True
    ).count()
    incorrect_count = total_feedback - correct_count

    # Training data stats
    training_service = TrainingDataService(db)
    training_stats = training_service.get_training_stats()

    return {
        "total_feedback": total_feedback,
        "correct_confirmations": correct_count,
        "corrections": incorrect_count,
        "accuracy_rate": correct_count / total_feedback if total_feedback > 0 else 0,
        "training_data": training_stats,
    }
