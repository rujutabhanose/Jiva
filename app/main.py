from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import warnings
from app.api.v1.router import api_router
from app.core.config import settings
from app.db.init_db import init_db
from app.services.image_utils import cleanup_old_images, UPLOAD_DIR

# Suppress bcrypt version warning (harmless compatibility message)
warnings.filterwarnings("ignore", message=".*bcrypt.*")


async def cleanup_images_periodically():
    """Background task to clean up old images every 24 hours"""
    while True:
        try:
            await asyncio.sleep(86400)  # 24 hours in seconds
            print("🧹 Running scheduled image cleanup...")
            deleted_count = cleanup_old_images()
            print(f"✅ Cleanup complete: {deleted_count} images deleted")
        except Exception as e:
            print(f"⚠️  Error in scheduled cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the application.
    Initialize database tables and ML models on startup.
    """
    # Startup: Initialize database
    print("🔧 Initializing database...")
    init_db()
    print("✅ Database initialized successfully")

    # Startup: Load ML model
    print("🤖 Loading plant identification model...")
    try:
        from app.services.plant_identifier import load_model
        load_model()
        print("✅ Plant identification model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Failed to load model: {e}")
        print("   The /api/v1/identify endpoint will not work until the model is loaded.")

    # Startup: Run initial cleanup
    print("🧹 Running initial image cleanup...")
    try:
        deleted_count = cleanup_old_images()
        print(f"✅ Initial cleanup complete: {deleted_count} old images deleted")
    except Exception as e:
        print(f"⚠️  Warning: Failed to run initial cleanup: {e}")

    # Startup: Launch background cleanup task
    cleanup_task = asyncio.create_task(cleanup_images_periodically())

    yield

    # Shutdown: Cancel background tasks
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("👋 Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Jiva Plants API - Backend for plant care management",
    lifespan=lifespan
)

# CORS middleware configuration for mobile app support
if settings.CORS_ALLOW_ALL_ORIGINS and settings.ENVIRONMENT == "development":
    # Allow all origins in development for easier mobile testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Must be False when allow_origins is ["*"]
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
        max_age=settings.CORS_MAX_AGE,
    )
else:
    # Production: Use specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
        max_age=settings.CORS_MAX_AGE,
    )

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Mount static files for serving uploaded images
# Create uploads directory if it doesn't exist
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Welcome to Jiva Plants API",
        "version": settings.VERSION,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "jiva-plants-api"}


@app.get("/admin/storage/stats")
async def storage_stats():
    """Get image storage statistics (admin endpoint)"""
    from app.services.image_utils import get_storage_stats
    stats = get_storage_stats()
    stats["upload_directory"] = str(UPLOAD_DIR)
    stats["retention_days"] = 30
    return stats


@app.post("/admin/storage/cleanup")
async def manual_cleanup(days: int = 30):
    """
    Manually trigger cleanup of old images (admin endpoint)

    Args:
        days: Delete images older than this many days (default: 30)
    """
    from app.services.image_utils import cleanup_old_images
    deleted_count = cleanup_old_images(days=days)
    return {
        "success": True,
        "deleted_count": deleted_count,
        "retention_days": days,
        "message": f"Deleted {deleted_count} images older than {days} days"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
