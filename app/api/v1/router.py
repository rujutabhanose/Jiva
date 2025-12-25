from fastapi import APIRouter
from app.api.v1 import users, auth, scans, identify, diagnose

api_router = APIRouter()

# Include individual route modules
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(scans.router, prefix="/scans", tags=["Scans"])
api_router.include_router(identify.router, prefix="/identify", tags=["Identify"])
api_router.include_router(diagnose.router, prefix="/diagnose", tags=["Diagnose"])
