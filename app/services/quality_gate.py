"""
Image quality gate - checks if image is suitable for analysis
"""

def image_quality_check(image_path: str) -> bool:
    """
    Check if image meets quality requirements for plant identification.

    Args:
        image_path: Path to the image file

    Returns:
        True if image passes quality check, False otherwise
    """
    from pathlib import Path
    import os

    path = Path(image_path)

    # Basic checks
    if not path.exists():
        return False

    # Check file size (at least 10KB, at most 50MB)
    file_size = os.path.getsize(path)
    if file_size < 10 * 1024:  # 10KB minimum
        return False
    if file_size > 50 * 1024 * 1024:  # 50MB maximum
        return False

    # Check extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.heic'}
    if path.suffix.lower() not in valid_extensions:
        return False

    return True
