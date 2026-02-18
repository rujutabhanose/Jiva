"""
Model Scheduler Service

Handles scheduled model retraining using APScheduler.
Runs weekly checks and triggers retraining when sufficient data is available.

On lightweight deployments (tflite-runtime only), the scheduler is disabled
and retraining happens via GitHub Actions instead.
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# APScheduler + training are optional (not needed on tflite-runtime deployments)
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False
    logger.info("APScheduler not available — scheduled training disabled (use GitHub Actions)")

from app.db.session import SessionLocal
from app.services.training_data_service import TrainingDataService
from app.services.model_trainer import run_training_job
from app.core.config import settings

# Global scheduler instance
scheduler = None


async def weekly_training_check():
    """
    Weekly scheduled job to check and trigger model retraining.

    Runs every Sunday at 2:00 AM UTC.
    """
    logger.info("Starting weekly training check...")

    db: Session = SessionLocal()
    try:
        training_service = TrainingDataService(db)

        # Check each model type
        for model_type in ["coleaf", "disease"]:
            if training_service.can_retrain(model_type):
                unused_count = len(training_service.get_unused_samples(model_type))
                logger.info(
                    f"Triggering {model_type} retraining with {unused_count} samples"
                )

                # Run training
                result = await run_training_job(
                    db=db,
                    model_type=model_type,
                    epochs=10,
                    auto_activate=True,  # Auto-activate if accuracy is good
                )

                if result["success"]:
                    logger.info(f"Training completed: {result['message']}")
                    if result["metrics"]:
                        logger.info(f"Metrics: {result['metrics']}")
                else:
                    logger.warning(f"Training failed: {result['message']}")
            else:
                stats = training_service.get_training_stats()
                model_stats = stats["by_model_type"].get(model_type, {})
                unused = model_stats.get("unused", 0)
                logger.info(
                    f"Skipping {model_type}: only {unused}/{settings.MIN_TRAINING_SAMPLES} samples"
                )

    except Exception as e:
        logger.error(f"Weekly training check failed: {e}", exc_info=True)
    finally:
        db.close()


async def check_and_train_model(model_type: str) -> dict:
    """
    Check if a specific model can be retrained and trigger if possible.

    Args:
        model_type: Type of model to check ("coleaf", "disease", "plant_id")

    Returns:
        Dictionary with result status
    """
    db: Session = SessionLocal()
    try:
        training_service = TrainingDataService(db)

        if not training_service.can_retrain(model_type):
            stats = training_service.get_training_stats()
            model_stats = stats["by_model_type"].get(model_type, {})
            return {
                "success": False,
                "message": f"Insufficient samples: {model_stats.get('unused', 0)}/{settings.MIN_TRAINING_SAMPLES}",
                "can_train": False,
            }

        result = await run_training_job(
            db=db,
            model_type=model_type,
            epochs=10,
            auto_activate=True,
        )

        return {
            "success": result["success"],
            "message": result["message"],
            "metrics": result.get("metrics"),
            "can_train": True,
        }

    except Exception as e:
        logger.error(f"Training check failed for {model_type}: {e}")
        return {
            "success": False,
            "message": str(e),
            "can_train": False,
        }
    finally:
        db.close()


def init_scheduler():
    """
    Initialize the APScheduler with weekly training job.

    Should be called during application startup.
    """
    global scheduler

    if not _HAS_SCHEDULER:
        logger.info("Scheduler not available — retraining handled by GitHub Actions")
        return

    if scheduler is not None:
        logger.warning("Scheduler already initialized")
        return

    scheduler = AsyncIOScheduler()

    # Schedule weekly training check
    # Runs every Sunday at 2:00 AM UTC
    scheduler.add_job(
        weekly_training_check,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_training",
        name="Weekly Model Training Check",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Model training scheduler initialized (weekly on Sundays at 2:00 AM UTC)")


def shutdown_scheduler():
    """
    Shutdown the scheduler gracefully.

    Should be called during application shutdown.
    """
    global scheduler

    if scheduler is not None:
        scheduler.shutdown(wait=True)
        scheduler = None
        logger.info("Model training scheduler shut down")


def get_scheduler_status() -> dict:
    """
    Get the current status of the scheduler.

    Returns:
        Dictionary with scheduler status info
    """
    global scheduler

    if scheduler is None:
        return {
            "running": False,
            "jobs": [],
        }

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


def trigger_immediate_training(model_type: str) -> dict:
    """
    Trigger an immediate training job (outside of schedule).

    Args:
        model_type: Type of model to train

    Returns:
        Dictionary with job info
    """
    global scheduler

    if scheduler is None:
        return {
            "success": False,
            "message": "Scheduler not initialized — training handled by GitHub Actions",
        }

    # Add a one-time job
    job = scheduler.add_job(
        check_and_train_model,
        args=[model_type],
        id=f"immediate_training_{model_type}_{datetime.utcnow().timestamp()}",
        name=f"Immediate {model_type} Training",
    )

    return {
        "success": True,
        "message": f"Training job scheduled for {model_type}",
        "job_id": job.id,
    }
