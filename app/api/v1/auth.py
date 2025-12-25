from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.user import User
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
)

router = APIRouter()


def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    country: Optional[str] = None
    userType: Optional[str] = None  # 'Home gardener' | 'Nursery' | 'Farmer' | 'Other'
    plantTypes: Optional[List[str]] = None  # Array of plant types
    device_id: Optional[str] = None  # Device identifier from mobile app
    platform: Optional[str] = None  # 'ios' or 'android'


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User login endpoint"""
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create access token
    access_token = create_access_token(subject=user.email)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "country": user.country,
            "userType": user.user_type,
            "plantTypes": user.plant_types,
            "isPremium": user.is_premium,
            "freeScansLeft": user.free_scans_left
        }
    }


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    device_user = None
    if request.device_id:
        device_user = db.query(User).filter(User.device_id == request.device_id).first()

    if device_user:
        device_user.email = request.email
        device_user.hashed_password = hash_password(request.password)
        device_user.name = request.name
        device_user.country = request.country
        device_user.user_type = request.userType
        device_user.plant_types = request.plantTypes
        device_user.platform = request.platform
        device_user.is_verified = True

        db.commit()
        db.refresh(device_user)
        user = device_user
    else:
        user = User(
            email=request.email,
            hashed_password=hash_password(request.password),
            name=request.name,
            country=request.country,
            user_type=request.userType,
            plant_types=request.plantTypes,
            device_id=request.device_id,
            platform=request.platform,
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create access token
    access_token = create_access_token(subject=user.email)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "country": user.country,
            "userType": user.user_type,
            "plantTypes": user.plant_types,
            "isPremium": user.is_premium,
            "freeScansLeft": user.free_scans_left
        }
    }

@router.post("/logout")
async def logout():
    """User logout endpoint"""
    # In a stateless JWT system, logout is handled client-side
    # by removing the token from storage
    return {"message": "Logged out successfully"}
