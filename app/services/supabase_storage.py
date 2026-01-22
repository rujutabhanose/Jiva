"""
Supabase Storage Service for Training Data

Handles uploading, downloading, and managing training images
in Supabase Storage (free tier: 1GB).

Storage structure:
    training-data/
    ├── coleaf/
    │   ├── nitrogen_deficiency/
    │   └── ...
    ├── disease/
    │   ├── powdery_mildew/
    │   └── ...
    └── plant_id/
        ├── tomato/
        └── ...
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import httpx
import base64

from app.core.config import settings

logger = logging.getLogger(__name__)


class SupabaseStorageService:
    """Service for interacting with Supabase Storage."""

    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.bucket = settings.SUPABASE_BUCKET
        self._initialized = False

        if self.url and self.key:
            self._initialized = True
            logger.info(f"Supabase Storage initialized for bucket: {self.bucket}")
        else:
            logger.warning(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY in .env for cloud storage."
            )

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return self._initialized

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for Supabase API."""
        return {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key,
        }

    def _get_storage_url(self, path: str = "") -> str:
        """Build Supabase storage URL."""
        base = f"{self.url}/storage/v1/object/{self.bucket}"
        if path:
            return f"{base}/{path}"
        return base

    async def upload_training_image(
        self,
        image_path: str,
        model_type: str,
        label: str,
        scan_id: int,
    ) -> Optional[str]:
        """
        Upload an image to Supabase Storage for training.

        Args:
            image_path: Local path to the image file
            model_type: Type of model (coleaf, disease, plant_id)
            label: Confirmed label (e.g., nitrogen_deficiency, powdery_mildew)
            scan_id: Associated scan ID

        Returns:
            Public URL of uploaded image, or None if upload failed
        """
        if not self.is_configured:
            logger.warning("Supabase not configured, skipping upload")
            return None

        try:
            # Read image file
            image_file = Path(image_path)
            if not image_file.exists():
                logger.error(f"Image file not found: {image_path}")
                return None

            # Generate storage path: {model_type}/{label}/{timestamp}_{scan_id}.jpg
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            extension = image_file.suffix or ".jpg"
            storage_path = f"{model_type}/{label}/{timestamp}_{scan_id}{extension}"

            # Read file content
            with open(image_path, "rb") as f:
                file_content = f.read()

            # Determine content type
            content_type = "image/jpeg"
            if extension.lower() == ".png":
                content_type = "image/png"

            # Upload to Supabase
            upload_url = self._get_storage_url(storage_path)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    upload_url,
                    headers={
                        **self._get_headers(),
                        "Content-Type": content_type,
                    },
                    content=file_content,
                )

                if response.status_code in (200, 201):
                    # Return public URL
                    public_url = f"{self.url}/storage/v1/object/public/{self.bucket}/{storage_path}"
                    logger.info(f"Uploaded training image: {storage_path}")
                    return public_url
                else:
                    logger.error(
                        f"Failed to upload image: {response.status_code} - {response.text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error uploading training image: {e}")
            return None

    async def upload_training_image_from_base64(
        self,
        base64_data: str,
        model_type: str,
        label: str,
        scan_id: int,
    ) -> Optional[str]:
        """
        Upload a base64-encoded image to Supabase Storage.

        Args:
            base64_data: Base64-encoded image data (with or without data URI prefix)
            model_type: Type of model (coleaf, disease, plant_id)
            label: Confirmed label
            scan_id: Associated scan ID

        Returns:
            Public URL of uploaded image, or None if upload failed
        """
        if not self.is_configured:
            logger.warning("Supabase not configured, skipping upload")
            return None

        try:
            # Remove data URI prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]

            # Decode base64
            file_content = base64.b64decode(base64_data)

            # Generate storage path
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            storage_path = f"{model_type}/{label}/{timestamp}_{scan_id}.jpg"

            # Upload to Supabase
            upload_url = self._get_storage_url(storage_path)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    upload_url,
                    headers={
                        **self._get_headers(),
                        "Content-Type": "image/jpeg",
                    },
                    content=file_content,
                )

                if response.status_code in (200, 201):
                    public_url = f"{self.url}/storage/v1/object/public/{self.bucket}/{storage_path}"
                    logger.info(f"Uploaded training image from base64: {storage_path}")
                    return public_url
                else:
                    logger.error(
                        f"Failed to upload base64 image: {response.status_code} - {response.text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error uploading base64 training image: {e}")
            return None

    async def download_training_image(self, storage_path: str) -> Optional[bytes]:
        """
        Download an image from Supabase Storage.

        Args:
            storage_path: Path within the bucket (e.g., "coleaf/nitrogen_deficiency/xxx.jpg")

        Returns:
            Image bytes, or None if download failed
        """
        if not self.is_configured:
            return None

        try:
            download_url = self._get_storage_url(storage_path)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    download_url,
                    headers=self._get_headers(),
                )

                if response.status_code == 200:
                    return response.content
                else:
                    logger.error(f"Failed to download image: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error downloading training image: {e}")
            return None

    async def list_training_images(
        self, model_type: str, label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List training images for a model type and optionally a specific label.

        Args:
            model_type: Type of model (coleaf, disease, plant_id)
            label: Optional specific label to filter by

        Returns:
            List of file metadata dictionaries
        """
        if not self.is_configured:
            return []

        try:
            prefix = f"{model_type}/"
            if label:
                prefix = f"{model_type}/{label}/"

            list_url = f"{self.url}/storage/v1/object/list/{self.bucket}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    list_url,
                    headers=self._get_headers(),
                    json={"prefix": prefix},
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to list images: {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"Error listing training images: {e}")
            return []

    async def delete_training_image(self, storage_path: str) -> bool:
        """
        Delete an image from Supabase Storage.

        Args:
            storage_path: Path within the bucket

        Returns:
            True if deletion successful, False otherwise
        """
        if not self.is_configured:
            return False

        try:
            delete_url = self._get_storage_url(storage_path)

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    delete_url,
                    headers=self._get_headers(),
                )

                if response.status_code in (200, 204):
                    logger.info(f"Deleted training image: {storage_path}")
                    return True
                else:
                    logger.error(f"Failed to delete image: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error deleting training image: {e}")
            return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for the training data bucket.

        Returns:
            Dictionary with storage stats per model type and label
        """
        stats = {
            "total_images": 0,
            "by_model_type": {},
        }

        if not self.is_configured:
            return stats

        for model_type in ["coleaf", "disease", "plant_id"]:
            images = await self.list_training_images(model_type)
            model_stats = {"total": len(images), "by_label": {}}

            # Group by label
            for img in images:
                name = img.get("name", "")
                parts = name.split("/")
                if len(parts) >= 2:
                    label = parts[1]
                    model_stats["by_label"][label] = model_stats["by_label"].get(label, 0) + 1

            stats["by_model_type"][model_type] = model_stats
            stats["total_images"] += model_stats["total"]

        return stats


# Global instance
supabase_storage = SupabaseStorageService()
