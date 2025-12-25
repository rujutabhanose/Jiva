# Image Storage Management System

## Overview

The Jiva backend now includes a comprehensive image storage management system with automatic compression, size validation, cleanup, and CDN-ready serving capabilities.

## Features

### 1. **Automatic Image Compression** ✅
- **Quality**: JPEG compression at 85% quality
- **Resizing**: Images larger than 2048px are automatically resized (maintains aspect ratio)
- **Format Conversion**: All images converted to RGB JPEG for consistency
- **Optimization**: Uses PIL's optimize flag for additional compression

**Example Savings:**
```
Original: 5.2 MB (4032x3024 PNG)
↓
Compressed: 1.1 MB (2048x1536 JPEG)
= 79% size reduction
```

### 2. **File Size Validation** ✅
- **Maximum Upload**: 10 MB per image
- **HTTP 413** returned if file exceeds limit
- Pre-compression size check prevents unnecessary processing

### 3. **Automatic Cleanup** ✅
- **Retention Period**: 30 days (configurable)
- **Schedule**: Runs every 24 hours automatically
- **On Startup**: Initial cleanup on app startup
- **Logging**: Detailed logs of cleanup operations

### 4. **Image Serving** ✅
- **Static Files**: Mounted at `/uploads/` endpoint
- **Direct URLs**: `http://yourserver/uploads/abc123.jpg`
- **CDN Ready**: Can easily point CDN to `/uploads` directory

## Configuration

Edit `/backend/app/services/image_utils.py`:

```python
MAX_FILE_SIZE_MB = 10          # Maximum upload size
MAX_IMAGE_DIMENSION = 2048     # Max width/height
JPEG_QUALITY = 85              # Compression quality (0-100)
IMAGE_RETENTION_DAYS = 30      # Days to keep images
```

## API Endpoints

### Upload Image (via diagnosis)
```bash
POST /api/v1/diagnose
Content-Type: multipart/form-data

file: <image file>
device_id: <user device id>
```

**Response includes:**
```json
{
  "image_url": "/uploads/abc123.jpg",
  ...
}
```

### Serve Image
```bash
GET /uploads/abc123.jpg
```
Returns the image file directly.

### Storage Statistics (Admin)
```bash
GET /admin/storage/stats
```

**Response:**
```json
{
  "total_images": 1523,
  "total_size_mb": 342.5,
  "oldest_image_days": 28,
  "newest_image_days": 0,
  "upload_directory": "/path/to/uploads",
  "retention_days": 30
}
```

### Manual Cleanup (Admin)
```bash
POST /admin/storage/cleanup?days=30
```

**Response:**
```json
{
  "success": true,
  "deleted_count": 45,
  "retention_days": 30,
  "message": "Deleted 45 images older than 30 days"
}
```

## CDN Setup (Optional)

### Option 1: Nginx Reverse Proxy
```nginx
location /uploads/ {
    alias /path/to/jiva/backend/uploads/;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

### Option 2: CloudFlare / AWS CloudFront
1. Point CDN origin to `https://yourserver/uploads`
2. Set cache TTL to 30 days
3. Update frontend to use CDN URL

### Option 3: S3/Cloud Storage (Future)
For production at scale, consider migrating to:
- AWS S3 + CloudFront
- Google Cloud Storage + CDN
- Azure Blob Storage + CDN

**Migration Steps:**
1. Update `save_image()` to upload to cloud storage
2. Return cloud URL instead of local path
3. Set up lifecycle rules for automatic deletion
4. Point CloudFront/CDN to cloud bucket

## Monitoring

### Check Storage Stats
```bash
curl http://localhost:8000/admin/storage/stats
```

### Logs
Search for cleanup operations:
```bash
grep "Cleanup complete" backend/logs/app.log
grep "Image saved" backend/logs/app.log | tail -20
```

### Disk Usage
```bash
du -sh uploads/
ls -lh uploads/ | wc -l  # Count images
```

## Troubleshooting

### Images not being cleaned up
1. Check logs for cleanup errors
2. Verify file permissions on `uploads/` directory
3. Manually trigger cleanup: `POST /admin/storage/cleanup`

### High disk usage
1. Check storage stats: `GET /admin/storage/stats`
2. Reduce retention period in config
3. Run aggressive cleanup: `POST /admin/storage/cleanup?days=7`

### Upload failures
1. Check file size (max 10MB)
2. Verify image format (should be JPEG, PNG, or common format)
3. Check disk space: `df -h`

## Security Considerations

### Current Implementation
- ✅ File size limits prevent DOS attacks
- ✅ Image validation prevents malicious uploads
- ✅ Auto-cleanup prevents disk filling
- ⚠️  Admin endpoints have no authentication (TODO)
- ⚠️  Public image URLs (no access control)

### Production Recommendations
1. **Add Authentication to Admin Endpoints**
   ```python
   @app.get("/admin/storage/stats", dependencies=[Depends(verify_admin)])
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)

   @app.post("/api/v1/diagnose")
   @limiter.limit("10/minute")
   async def diagnose_plant(...):
   ```

3. **Signed URLs for Sensitive Images**
   If images should be private, generate time-limited signed URLs

4. **Content Security**
   - Validate image content (not just extension)
   - Scan for malware in uploads
   - Strip EXIF metadata for privacy

## Performance

### Current Performance
- **Upload + Compression**: ~200-500ms per image
- **Cleanup**: ~1-2ms per image deleted
- **Serving**: Direct static file (nginx/fast)

### Optimization Tips
1. **Enable Nginx Caching**
   ```nginx
   proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=img_cache:10m;
   ```

2. **Use CDN**
   Offload 90%+ of image serving traffic

3. **Async Processing**
   For very large uploads, process compression in background task

4. **WebP Format**
   Consider converting to WebP for 25-35% additional savings
   (Requires client-side support check)

## Metrics to Track

1. **Storage Growth Rate**: Images/day, MB/day
2. **Cleanup Efficiency**: Images deleted/day
3. **Compression Ratio**: Average size reduction
4. **Upload Success Rate**: Failed uploads / total uploads
5. **CDN Hit Rate**: If using CDN

## Future Enhancements

- [ ] WebP format support for modern browsers
- [ ] Automatic thumbnail generation
- [ ] Image deduplication (hash-based)
- [ ] Cloud storage integration (S3, GCS)
- [ ] Admin dashboard for storage management
- [ ] Prometheus metrics export
- [ ] User-specific image quotas
