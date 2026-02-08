from fastapi import APIRouter
from app.api.v1 import users, auth, scans, identify, diagnose, feedback, admin_training

# Import enhanced plant identification router
import sys
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_root))
from scripts.plant_id.plant_identifier_enhanced_api import plant_id_router

api_router = APIRouter()

# Include individual route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(scans.router, prefix="/scans", tags=["Scans"])
api_router.include_router(identify.router, prefix="/identify", tags=["Identify"])
api_router.include_router(diagnose.router, prefix="/diagnose", tags=["Diagnose"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

# Admin endpoints for continuous learning
api_router.include_router(admin_training.router, prefix="/admin", tags=["Admin - Training"])

# Enhanced plant identification with ensemble models
api_router.include_router(plant_id_router, tags=["Enhanced Plant ID"])
