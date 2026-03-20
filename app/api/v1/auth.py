from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import random
import string
from app.db.session import SessionLocal
from app.models.user import User
from app.models.password_reset import PasswordResetToken
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from app.core.email import send_password_reset_otp
from app.services.geolocation import get_country_from_ip, extract_client_ip

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
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    reset_token: str
    new_password: str


class MessageResponse(BaseModel):
    message: str


class VerifyOTPResponse(BaseModel):
    valid: bool
    reset_token: str


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

    # Create tokens
    access_token = create_access_token(subject=user.email)
    refresh_token = create_refresh_token(subject=user.email)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
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
async def register(request: RegisterRequest, http_request: Request, db: Session = Depends(get_db)):
    """Register a new user. All users start with 1 free diagnosis scan."""

    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Geolocate the client IP to verify India users — not the user-selected country
    client_ip = extract_client_ip(http_request)
    ip_country = await get_country_from_ip(client_ip) if client_ip else None
    india_free_expires_at = (
        datetime.utcnow() + timedelta(days=183)  # ~6 months
        if ip_country == "IN"
        else None
    )

    # Create new user - starts with 1 free scan (free_scans_left defaults to 1)
    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        name=request.name,
        country=request.country,
        user_type=request.userType,
        plant_types=request.plantTypes,
        device_id=request.device_id,  # Keep for analytics
        platform=request.platform,
        is_verified=True,
        registration_ip=client_ip,
        ip_country=ip_country,
        india_free_expires_at=india_free_expires_at,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create tokens
    access_token = create_access_token(subject=user.email)
    refresh_token = create_refresh_token(subject=user.email)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
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


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new access token + rotated refresh token"""
    email = verify_refresh_token(request.refresh_token)

    # Ensure the user still exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        from fastapi import status as http_status
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail="User not found")

    new_access_token = create_access_token(subject=email)
    new_refresh_token = create_refresh_token(subject=email)

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Request password reset OTP.
    Sends a 6-digit OTP to the user's email if the account exists.
    """
    # Check if user exists
    user = db.query(User).filter(User.email == request.email).first()

    # Always return success to prevent email enumeration attacks
    if not user:
        return {"message": "If an account exists with this email, you will receive a reset code."}

    # Invalidate any existing OTPs for this email
    db.query(PasswordResetToken).filter(
        PasswordResetToken.email == request.email,
        PasswordResetToken.used == False
    ).update({"used": True})
    db.commit()

    # Generate new OTP
    otp = generate_otp()

    # Create new token record
    token_record = PasswordResetToken.create_otp(email=request.email, otp=otp)
    db.add(token_record)
    db.commit()

    # Send OTP via email
    email_sent = send_password_reset_otp(email=request.email, otp=otp, name=user.name)
    if not email_sent:
        raise HTTPException(status_code=500, detail="Failed to send reset email. Please try again.")

    return {"message": "If an account exists with this email, you will receive a reset code."}


@router.post("/verify-reset-otp", response_model=VerifyOTPResponse)
async def verify_reset_otp(request: VerifyOTPRequest, db: Session = Depends(get_db)):
    """
    Verify the OTP code.
    Returns a reset_token if the OTP is valid.
    """
    # Find the most recent valid OTP for this email
    token_record = db.query(PasswordResetToken).filter(
        PasswordResetToken.email == request.email,
        PasswordResetToken.otp == request.otp,
        PasswordResetToken.used == False
    ).order_by(PasswordResetToken.created_at.desc()).first()

    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    if not token_record.is_valid():
        raise HTTPException(status_code=400, detail="Code has expired. Please request a new one.")

    # Generate a secure reset token
    reset_token = secrets.token_urlsafe(32)
    token_record.reset_token = reset_token
    db.commit()

    return {"valid": True, "reset_token": reset_token}


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password using the reset_token from OTP verification.
    """
    # Find the token record
    token_record = db.query(PasswordResetToken).filter(
        PasswordResetToken.email == request.email,
        PasswordResetToken.reset_token == request.reset_token,
        PasswordResetToken.used == False
    ).first()

    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid reset token")

    if not token_record.is_valid():
        raise HTTPException(status_code=400, detail="Reset token has expired. Please start over.")

    # Find the user
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate password length
    if len(request.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Update password
    user.hashed_password = hash_password(request.new_password)

    # Mark token as used
    token_record.used = True

    db.commit()

    return {"message": "Password reset successfully. You can now sign in with your new password."}
