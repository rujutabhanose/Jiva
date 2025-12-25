"""
Image Upload Utilities for Jiva Diagnosis
Handles temp file storage for model processing with auto-cleanup
Includes compression, size limits, and automatic cleanup
"""

import uuid
import tempfile
import os
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from typing import Tuple, Optional
import logging
from PIL import Image
from datetime import datetime, timedelta
import io

logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Image constraints
MAX_FILE_SIZE_MB = 10  # Maximum upload size in MB
MAX_IMAGE_DIMENSION = 2048  # Maximum width/height in pixels
JPEG_QUALITY = 85  # Compression quality (0-100)
IMAGE_RETENTION_DAYS = 30  # Keep images for 30 days

_temp_files = []  # Track temp files for cleanup

def save_image(file: UploadFile) -> str:
    """
    Save uploaded image to persistent uploads directory with compression and validation

    Args:
        file: FastAPI UploadFile

    Returns:
        Full path to saved image

    Raises:
        HTTPException: If file is too large or invalid format
    """
    # Read file content
    content = file.file.read()
    file.file.seek(0)  # Reset file pointer for potential re-use

    # Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"File too large: {file_size_mb:.2f}MB (max: {MAX_FILE_SIZE_MB}MB)")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
        )

    try:
        # Open image with PIL
        img = Image.open(io.BytesIO(content))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if too large (maintain aspect ratio)
        if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
            logger.info(f"Image resized from {img.size} to fit within {MAX_IMAGE_DIMENSION}px")

        # Generate unique filename
        filename = f"{uuid.uuid4()}.jpg"
        path = UPLOAD_DIR / filename

        # Save with compression
        img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)

        # Log compression stats
        compressed_size_mb = path.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - compressed_size_mb / file_size_mb) * 100 if file_size_mb > 0 else 0
        logger.info(f"Image saved: {path} ({compressed_size_mb:.2f}MB, {compression_ratio:.1f}% reduction)")

        return str(path)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

def save_temp_image(image_bytes: bytes) -> str:
    """
    Save image bytes to temporary file for model processing
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Temporary file path (auto-cleaned after use)
    """
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    
    try:
        with open(path, 'wb') as f:
            f.write(image_bytes)
        _temp_files.append(path)
        logger.debug(f"Temp image created: {path}")
        return path
    except Exception as e:
        os.unlink(path)
        raise e

def cleanup_temp_image(image_path: str):
    """
    Clean up temporary image file after model processing
    """
    global _temp_files
    try:
        if image_path in _temp_files:
            os.unlink(image_path)
            _temp_files.remove(image_path)
            logger.debug(f"Temp image cleaned: {image_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {image_path}: {e}")

def cleanup_all_temp_files():
    """
    Clean up all temporary files (call on app shutdown)
    """
    global _temp_files
    for path in _temp_files[:]:  # Copy to avoid modification during iteration
        cleanup_temp_image(path)

def get_image_url(image_path: str) -> str:
    """
    Convert local image path to public URL (for serving images)

    Args:
        image_path: Local path like "uploads/abc123.jpg"

    Returns:
        Public URL or None if not servable
    """
    try:
        path = Path(image_path).resolve()
        upload_dir = UPLOAD_DIR.resolve()

        # Check if path is within uploads directory
        path.relative_to(upload_dir)
        return f"/uploads/{path.name}"
    except (ValueError, Exception):
        # Path is not relative to upload_dir
        return None

def cleanup_old_images(days: int = IMAGE_RETENTION_DAYS) -> int:
    """
    Remove uploaded images older than specified days

    Args:
        days: Number of days to retain images (default: IMAGE_RETENTION_DAYS)

    Returns:
        Number of images deleted
    """
    if not UPLOAD_DIR.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(days=days)
    deleted_count = 0
    total_size_freed = 0

    try:
        for image_file in UPLOAD_DIR.glob("*.jpg"):
            # Check file modification time
            file_mtime = datetime.fromtimestamp(image_file.stat().st_mtime)

            if file_mtime < cutoff_time:
                file_size = image_file.stat().st_size
                try:
                    image_file.unlink()
                    deleted_count += 1
                    total_size_freed += file_size
                    logger.debug(f"Deleted old image: {image_file.name} (age: {(datetime.now() - file_mtime).days} days)")
                except Exception as e:
                    logger.warning(f"Failed to delete {image_file}: {e}")

        if deleted_count > 0:
            size_mb = total_size_freed / (1024 * 1024)
            logger.info(f"Cleanup complete: Deleted {deleted_count} images, freed {size_mb:.2f}MB")
        else:
            logger.info(f"Cleanup complete: No images older than {days} days found")

        return deleted_count

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return deleted_count


def get_storage_stats() -> dict:
    """
    Get statistics about image storage

    Returns:
        Dictionary with storage statistics
    """
    if not UPLOAD_DIR.exists():
        return {
            "total_images": 0,
            "total_size_mb": 0,
            "oldest_image_days": 0,
            "newest_image_days": 0
        }

    total_size = 0
    image_files = list(UPLOAD_DIR.glob("*.jpg"))
    image_count = len(image_files)

    if image_count == 0:
        return {
            "total_images": 0,
            "total_size_mb": 0,
            "oldest_image_days": 0,
            "newest_image_days": 0
        }

    now = datetime.now()
    oldest_time = now
    newest_time = datetime.fromtimestamp(0)

    for image_file in image_files:
        total_size += image_file.stat().st_size
        file_mtime = datetime.fromtimestamp(image_file.stat().st_mtime)

        if file_mtime < oldest_time:
            oldest_time = file_mtime
        if file_mtime > newest_time:
            newest_time = file_mtime

    return {
        "total_images": image_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest_image_days": (now - oldest_time).days,
        "newest_image_days": (now - newest_time).days
    }


# Cleanup handler for app shutdown
import atexit
atexit.register(cleanup_all_temp_files)