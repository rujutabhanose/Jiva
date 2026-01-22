"""
Admin Training API

Endpoints for managing continuous learning:
- View training data statistics
- Trigger manual retraining
- Manage model versions
- View scheduler status
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
import logging

from app.api.deps import get_db, get_current_user
from app.models import user as user_model
from app.models.model_version import ModelVersion
from app.services.training_data_service import TrainingDataService
from app.services.model_manager import model_manager
from app.services.model_scheduler import (
    get_scheduler_status,
    trigger_immediate_training,
    check_and_train_model,
)
from app.services.model_trainer import run_training_job

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response models

class TriggerTrainingRequest(BaseModel):
    model_type: str  # "coleaf", "disease", "plant_id"
    epochs: int = 10
    auto_activate: bool = True


class ActivateVersionRequest(BaseModel):
    version_id: int


class ModelVersionResponse(BaseModel):
    id: int
    model_type: str
    version: str
    accuracy: Optional[float]
    validation_accuracy: Optional[float]
    training_samples_count: int
    is_active: bool
    is_baseline: bool
    created_at: str

    class Config:
        from_attributes = True


# Endpoints

@router.get("/training/stats")
async def get_training_stats(
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get statistics about collected training data.

    Returns counts of confirmed samples by model type and label.
    """
    training_service = TrainingDataService(db)
    stats = training_service.get_training_stats()

    return {
        "status": "success",
        "data": stats,
    }


@router.post("/training/trigger")
async def trigger_training(
    request: TriggerTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Manually trigger model retraining.

    Training runs in background and can be monitored via scheduler status.
    """
    if request.model_type not in ["coleaf", "disease", "plant_id"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}"
        )

    # Check if we have enough samples
    training_service = TrainingDataService(db)
    if not training_service.can_retrain(request.model_type):
        stats = training_service.get_training_stats()
        model_stats = stats["by_model_type"].get(request.model_type, {})
        return {
            "status": "insufficient_data",
            "message": f"Need at least 50 samples, have {model_stats.get('unused', 0)}",
            "current_samples": model_stats.get("unused", 0),
        }

    # Trigger training in background
    background_tasks.add_task(
        run_training_job,
        db=db,
        model_type=request.model_type,
        epochs=request.epochs,
        auto_activate=request.auto_activate,
    )

    logger.info(f"Training triggered for {request.model_type} by user {current_user.id}")

    return {
        "status": "training_started",
        "message": f"Training job started for {request.model_type}",
        "epochs": request.epochs,
        "auto_activate": request.auto_activate,
    }


@router.get("/models/versions")
async def list_model_versions(
    model_type: Optional[str] = None,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all model versions, optionally filtered by type.
    """
    query = db.query(ModelVersion).order_by(ModelVersion.created_at.desc())

    if model_type:
        query = query.filter(ModelVersion.model_type == model_type)

    versions = query.all()

    return {
        "status": "success",
        "data": [
            {
                "id": v.id,
                "model_type": v.model_type,
                "version": v.version,
                "accuracy": v.accuracy,
                "validation_accuracy": v.validation_accuracy,
                "training_samples_count": v.training_samples_count,
                "epochs_trained": v.epochs_trained,
                "is_active": v.is_active,
                "is_baseline": v.is_baseline,
                "notes": v.notes,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ],
    }


@router.get("/models/active")
async def get_active_models(
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get currently active model versions.
    """
    active_versions = {}

    for model_type in ["coleaf", "disease", "plant_id"]:
        version = model_manager.get_model_version(model_type)
        if version:
            active_versions[model_type] = {
                "id": version.id,
                "version": version.version,
                "accuracy": version.accuracy,
                "is_baseline": version.is_baseline,
            }
        else:
            active_versions[model_type] = {
                "id": None,
                "version": "baseline",
                "accuracy": None,
                "is_baseline": True,
            }

    return {
        "status": "success",
        "data": active_versions,
    }


@router.post("/models/activate/{version_id}")
async def activate_model_version(
    version_id: int,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Activate a specific model version (hot-swap).
    """
    # Get the version
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Hot-swap
    success, message = model_manager.hot_swap_model(db, version.model_type, version_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    logger.info(
        f"Model {version.model_type} activated to version {version.version} "
        f"by user {current_user.id}"
    )

    return {
        "status": "success",
        "message": message,
        "activated_version": version.version,
    }


@router.post("/models/rollback/{model_type}")
async def rollback_model(
    model_type: str,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Rollback a model to its previous version.
    """
    if model_type not in ["coleaf", "disease", "plant_id"]:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

    success, message = model_manager.rollback_model(db, model_type)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    logger.info(f"Model {model_type} rolled back by user {current_user.id}")

    return {
        "status": "success",
        "message": message,
    }


@router.get("/scheduler/status")
async def get_scheduler_status_endpoint(
    current_user: user_model.User = Depends(get_current_user),
):
    """
    Get the status of the training scheduler.
    """
    status = get_scheduler_status()

    return {
        "status": "success",
        "data": status,
    }


@router.get("/training/label-distribution/{model_type}")
async def get_label_distribution(
    model_type: str,
    current_user: user_model.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the distribution of labels in unused training data.

    Useful for understanding class imbalance before training.
    """
    if model_type not in ["coleaf", "disease", "plant_id"]:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

    training_service = TrainingDataService(db)
    distribution = training_service.get_label_distribution(model_type)

    return {
        "status": "success",
        "model_type": model_type,
        "distribution": distribution,
        "total_samples": sum(distribution.values()),
    }
