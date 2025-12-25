from typing import List, Union
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    PROJECT_NAME: str = "Jiva Plants API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production

    # CORS - Mobile App Support
    ALLOWED_ORIGINS: List[str] = [
        # Web development
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",

        # Expo development servers
        "http://localhost:8081",      # Expo Metro bundler
        "http://localhost:19000",     # Expo DevTools
        "http://localhost:19006",     # Expo web

        # Local network for mobile testing (update with your IP)
        "http://192.168.1.0/24",      # Local network range
        "exp://localhost:8081",        # Expo Go app

        # Mobile simulators/emulators
        "http://10.0.2.2:8000",       # Android emulator
        "http://127.0.0.1:8081",      # iOS simulator
    ]

    # Allow all origins in development (set to False in production)
    CORS_ALLOW_ALL_ORIGINS: bool = True

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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
