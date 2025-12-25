from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.user import User
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_email_verify_token,
)
from app.services.email_service import send_verification_email
from datetime import datetime

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
    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email to continue"
    )

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
        device_user.is_verified = False
        device_user.verified_at = None

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
            is_verified=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # ✅ Send verification email
    token = create_email_verify_token(user.id)
    send_verification_email(user.email, token)

    access_token = create_access_token(subject=user.email)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "isVerified": user.is_verified,
        }
    }

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    from app.core.security import decode_email_verify_token

    try:
        user_id = decode_email_verify_token(token)
    except Exception:
        return HTMLResponse(content="""
            <html>
            <head><title>Verification Failed</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #f44336;">Verification Failed</h1>
                <p>This verification link is invalid or has expired.</p>
                <p>Please request a new verification email from the app.</p>
            </body>
            </html>
        """, status_code=400)

    user = db.query(User).get(user_id)
    if not user:
        return HTMLResponse(content="""
            <html>
            <head><title>User Not Found</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #f44336;">User Not Found</h1>
                <p>The user associated with this verification link could not be found.</p>
            </body>
            </html>
        """, status_code=404)

    if not user.is_verified:
        user.is_verified = True
        user.verified_at = datetime.utcnow()
        db.commit()

    # Return HTML page with deep link
    deep_link = f"jivaplants://verify-email?token={token}"

    return HTMLResponse(content=f"""
        <html>
        <head>
            <title>Email Verified!</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script>
                // Try to open the app automatically
                window.location.href = "{deep_link}";

                // If app doesn't open in 2 seconds, show the manual button
                setTimeout(function() {{
                    document.getElementById('manual-open').style.display = 'block';
                }}, 2000);
            </script>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f5f5f5;">
            <div style="max-width: 500px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 64px; margin-bottom: 20px;">✅</div>
                <h1 style="color: #4CAF50; margin-bottom: 10px;">Email Verified!</h1>
                <p style="color: #666; font-size: 16px; margin-bottom: 30px;">
                    Your email has been successfully verified.
                </p>

                <div id="manual-open" style="display: none;">
                    <p style="color: #666; font-size: 14px; margin-bottom: 20px;">
                        If the app didn't open automatically, click the button below:
                    </p>
                    <a href="{deep_link}"
                       style="display: inline-block; background-color: #4CAF50; color: white; padding: 15px 30px;
                              text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">
                        Open Jiva Plants App
                    </a>

                    <p style="color: #999; font-size: 12px; margin-top: 30px;">
                        Testing on emulator? The app is already verified. Just sign in with your credentials.
                    </p>
                </div>
            </div>
        </body>
        </html>
    """)


@router.post("/resend-verification")
def resend_verification(request: LoginRequest, db: Session = Depends(get_db)):
    """Resend verification email to user"""
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()

    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a verification link has been sent"}

    if user.is_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    # Generate new verification token
    token = create_email_verify_token(user.id)

    # Send verification email
    try:
        send_verification_email(user.email, token)
        return {"message": "Verification email sent successfully"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to send verification email")

@router.post("/logout")
async def logout():
    """User logout endpoint"""
    # In a stateless JWT system, logout is handled client-side
    # by removing the token from storage
    return {"message": "Logged out successfully"}
