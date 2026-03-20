from typing import List, Union, Optional
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""

    PROJECT_NAME: str = "Jiva Plants API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production

    # CORS - Mobile App Support
    # NOTE: Cannot use ["*"] with credentials=True (JWT auth requires credentials)
    # Add your local IP address via EXTRA_ALLOWED_ORIGINS env variable
    # Example: EXTRA_ALLOWED_ORIGINS=http://192.168.0.121:8000,http://192.168.0.121:8081
    EXTRA_ALLOWED_ORIGINS: Optional[str] = None

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Build list of allowed origins from defaults + environment variable"""
        origins = [
            # Web development
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",

            # Expo development servers
            "http://localhost:8081",      # Expo Metro bundler
            "http://localhost:19000",     # Expo DevTools
            "http://localhost:19006",     # Expo web
            "http://127.0.0.1:8081",      # iOS simulator
            "http://127.0.0.1:19000",
            "http://127.0.0.1:19006",

            # Mobile emulators
            "http://10.0.2.2:8000",       # Android emulator
            "http://10.0.2.2:8081",
            "http://localhost:8000",      # iOS simulator
        ]

        # Add extra origins from environment variable
        if self.EXTRA_ALLOWED_ORIGINS:
            extra = [origin.strip() for origin in self.EXTRA_ALLOWED_ORIGINS.split(',')]
            origins.extend(extra)

        return origins

    # Deprecated: Cannot use True due to JWT authentication requirements
    # Kept for backwards compatibility but not used
    CORS_ALLOW_ALL_ORIGINS: bool = False

    # CORS headers
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = [
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token",
    ]
    CORS_EXPOSE_HEADERS: List[str] = ["Content-Range", "X-Content-Range"]
    CORS_MAX_AGE: int = 600  # 10 minutes

    # Database
    DATABASE_URL: str = "sqlite:///./jiva_plants.db"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # 15 minutes
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30  # 30 days

    # Email (Resend)
    # Get your API key from https://resend.com/api-keys
    RESEND_API_KEY: Optional[str] = "re_KZepRifa_Jii41rMftyRTiBKTcCXMVSyU"
    # For testing: use "onboarding@resend.dev" (works immediately, no domain verification needed)
    # For production: change to "noreply@jivaplants.com" after domain is verified at https://resend.com/domains
    EMAIL_FROM: str = "Jiva Plants <noreply@jivaplants.com>"

    # Supabase Storage (for training data - FREE tier: 1GB)
    SUPABASE_URL: Optional[str] = None  # e.g., "https://xxxxx.supabase.co"
    SUPABASE_KEY: Optional[str] = None  # Service role key for server-side operations
    SUPABASE_BUCKET: str = "training-data"  # Bucket name for training images

    # Continuous Learning Configuration
    MIN_TRAINING_SAMPLES: int = 50  # Minimum samples before retraining
    TRAINING_VALIDATION_SPLIT: float = 0.2  # 20% holdout for validation
    MAX_ACCURACY_DROP: float = 0.05  # Auto-rollback if accuracy drops > 5%

    # HuggingFace Configuration
    # Token for accessing gated models (juppy44/plant-identification-2m-vit-b)
    # Must have "Make calls to serverless Inference API" permission
    # Get your token from: https://huggingface.co/settings/tokens
    HUGGINGFACE_TOKEN: Optional[str] = None
    PLANTNET_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
