# Jiva Plants API - Deployment Guide

This guide covers deploying the Jiva Plants API to various platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Manual Deployment](#manual-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Production Checklist](#production-checklist)

## Prerequisites

- Python 3.11, 3.12, or 3.13 (NOT 3.14)
- Docker (optional, for containerized deployment)
- Git
- At least 4GB RAM (for ML models)
- 10GB disk space (for ML model caches)

## Environment Setup

### 1. Create Environment File

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

### 2. Generate Secure Secret Key

```bash
# Use OpenSSL to generate a secure secret key
openssl rand -hex 32
```

Update `.env` with the generated key:
```
SECRET_KEY=your-generated-key-here
```

### 3. Configure Database

**For Development (SQLite):**
```env
DATABASE_URL=sqlite:///./jiva_plants.db
```

**For Production (PostgreSQL recommended):**
```env
DATABASE_URL=postgresql://username:password@host:port/database
```

### 4. CORS Configuration

For production, specify allowed origins:
```env
CORS_ALLOW_ALL_ORIGINS=false
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## Docker Deployment

### Quick Start

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Custom Build

```bash
# Build image
docker build -t jiva-plants-api .

# Run container
docker run -d \
  -p 8000:8000 \
  --name jiva-api \
  --env-file .env \
  -v $(pwd)/uploads:/app/uploads \
  jiva-plants-api
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Manual Deployment

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database

The database will be automatically initialized on first run.

### 3. Run Application

**Development:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn with Uvicorn workers:
```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## Cloud Platform Deployment

### Render

1. Create a new Web Service
2. Connect your GitHub repository
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `.env.example`
5. Deploy

### Railway

1. Create new project from GitHub repo
2. Railway will auto-detect FastAPI
3. Add environment variables
4. Deploy

### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy jiva-plants-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS EC2

1. Launch an EC2 instance (Ubuntu 22.04 recommended)
2. SSH into the instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv nginx
```
4. Clone repository and setup
5. Configure Nginx as reverse proxy
6. Use systemd for process management

### Heroku

Create `Procfile`:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Deploy:
```bash
heroku create jiva-plants-api
heroku config:set SECRET_KEY=your-secret-key
git push heroku main
```

### DigitalOcean App Platform

1. Create new app from GitHub
2. Select repository
3. Configure environment variables
4. Deploy

## Production Checklist

### Security

- [ ] Generate and set a strong `SECRET_KEY`
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure specific CORS origins (disable `CORS_ALLOW_ALL_ORIGINS`)
- [ ] Use HTTPS/TLS certificates
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Review API authentication

### Database

- [ ] Switch from SQLite to PostgreSQL
- [ ] Set up regular database backups
- [ ] Configure connection pooling
- [ ] Set up database monitoring

### Performance

- [ ] Configure multiple workers (4-8 recommended)
- [ ] Set up caching (Redis recommended)
- [ ] Configure CDN for static files
- [ ] Optimize ML model loading
- [ ] Set up horizontal scaling if needed

### Monitoring

- [ ] Set up application logging
- [ ] Configure error tracking (Sentry, etc.)
- [ ] Set up uptime monitoring
- [ ] Configure performance monitoring
- [ ] Set up alerts for errors and downtime

### Storage

- [ ] Configure external storage for uploads (S3, CloudStorage)
- [ ] Set up automated image cleanup
- [ ] Monitor disk usage
- [ ] Configure backup strategy

### ML Models

- [ ] Pre-download ML models to avoid startup delays
- [ ] Consider model quantization for faster inference
- [ ] Set up GPU support if available
- [ ] Monitor model memory usage

### Environment Variables

Required for production:
```env
ENVIRONMENT=production
DATABASE_URL=postgresql://...
SECRET_KEY=<strong-secret-key>
CORS_ALLOW_ALL_ORIGINS=false
ALLOWED_ORIGINS=https://yourdomain.com
```

## Troubleshooting

### Model Loading Fails

ML models are large (several GB). First startup will download models:
- Plant Identification: ~500MB
- Disease Detection: ~15MB
- Ensure stable internet connection
- Allow 5-10 minutes for first startup

### Out of Memory

The application requires ~2-4GB RAM for ML models:
- Increase instance size
- Consider using CPU-optimized instances
- Use model quantization

### Slow API Responses

- Enable GPU if available
- Increase worker count
- Implement caching for common requests
- Consider using smaller/quantized models

### Database Connection Issues

For PostgreSQL:
- Verify connection string format
- Check firewall rules
- Ensure database is running
- Verify credentials

## Scaling

### Horizontal Scaling

- Use load balancer (Nginx, AWS ALB)
- Deploy multiple instances
- Shared database (PostgreSQL)
- Shared file storage (S3)

### Vertical Scaling

- Increase CPU cores for faster inference
- Add RAM for model caching
- Use GPU instances for ML workloads

## Support

For issues or questions:
- Check [API Documentation](API_ENDPOINTS.md)
- Review [README](README.md)
- Open an issue on GitHub

## License

See LICENSE file for details.
