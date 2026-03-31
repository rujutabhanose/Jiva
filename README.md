# Jiva Plants API

Backend API for the Jiva Plants mobile application - A plant care management system.

## Features

- User authentication (login, register, OTP verification, password reset)
- Plant identification via image upload (hybrid ML + external API)
- Plant disease diagnosis with treatment plans
- Scan history management with image storage (Supabase)
- User profile management with Pro/free tier enforcement
- Coupon code system for Pro access
- Diagnosis feedback collection and model retraining pipeline
- Geolocation-based free access (India free tier)
- Admin tools for training data and model management

## Tech Stack

- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database operations
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

## Setup

### Prerequisites

- Python 3.11, 3.12, or 3.13 (⚠️ **NOT 3.14** - pydantic-core doesn't support it yet)
- Poetry (recommended) or pip

### Installation

1. Create a virtual environment with Python 3.13:
```bash
# Create venv with Python 3.13
python3.13 -m venv .venv

# Activate venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Verify Python version (should be 3.13.x)
python --version
```

2. Install dependencies:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

Or with Poetry:
```bash
poetry install
```

2. Create a `.env` file (optional):
```env
DATABASE_URL=sqlite:///./jiva_plants.db
SECRET_KEY=your-secret-key
```

3. Run the development server:
```bash
poetry run python -m app.main
```

Or:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Mobile App Testing

### CORS Configuration

The backend is configured to support mobile app development:

**Development Mode** (default):
- Allows ALL origins for easier testing
- Perfect for Expo, React Native, and local development
- No CORS issues when testing on devices/emulators

**Production Mode**:
- Restricts to specific origins only
- Set `ENVIRONMENT=production` and `CORS_ALLOW_ALL_ORIGINS=false` in `.env`

### Testing with Mobile Device

1. **Find your local IP address**:
```bash
./get_local_ip.sh
```

2. **Start the backend server**:
```bash
python -m app.main
```

3. **Update your mobile app config**:
   - Use `http://YOUR_LOCAL_IP:8000/api/v1` as the API base URL
   - Example: `http://192.168.1.100:8000/api/v1`

4. **Make sure**:
   - Your phone and computer are on the same Wi-Fi network
   - Port 8000 is accessible (check firewall settings)

### Supported Origins (Development)

- Web: `localhost:3000`, `localhost:5173`, `localhost:8080`
- Expo: `localhost:8081`, `localhost:19000`, `localhost:19006`
- iOS Simulator: `127.0.0.1:8081`
- Android Emulator: `10.0.2.2:8000`
- Local Network: Your IP address (e.g., `192.168.1.x`)

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── deps.py                        # Shared dependencies (auth, DB session)
│   │   └── v1/
│   │       ├── auth.py                    # Login, register, OTP, password reset
│   │       ├── users.py                   # Profile, upgrade to Pro, cancel subscription
│   │       ├── scans.py                   # Scan CRUD + notes
│   │       ├── identify.py                # Plant identification endpoint
│   │       ├── diagnose.py                # Plant diagnosis endpoint
│   │       ├── feedback.py                # Diagnosis feedback collection
│   │       ├── admin_training.py          # Admin: training data & model management
│   │       └── router.py                  # Mounts all v1 routers
│   ├── core/
│   │   ├── config.py                      # App settings (env vars)
│   │   ├── security.py                    # JWT, password hashing
│   │   └── email.py                       # Email sending (OTP, reset)
│   ├── db/
│   │   ├── session.py                     # SQLAlchemy engine & session
│   │   ├── base.py                        # Base model import aggregator
│   │   └── init_db.py                     # DB initialisation
│   ├── models/
│   │   ├── user.py                        # User table
│   │   ├── scan.py                        # Scan table
│   │   ├── coupon.py                      # Coupon codes
│   │   ├── coupon_redemption.py           # Coupon redemption records
│   │   ├── diagnosis_feedback.py          # User feedback on diagnoses
│   │   ├── model_version.py               # ML model version tracking
│   │   ├── password_reset.py              # Password reset tokens
│   │   └── training_data.py               # Training data records
│   ├── services/
│   │   ├── diagnosis_engine.py            # Core disease diagnosis logic
│   │   ├── plant_identifier.py            # Plant identification (ML)
│   │   ├── hybrid_plant_identifier.py     # Hybrid ML + API identifier
│   │   ├── coleaf_engine.py               # CoLeaf model inference
│   │   ├── model_manager.py               # ML model loading & versioning
│   │   ├── model_trainer.py               # Model retraining pipeline
│   │   ├── model_scheduler.py             # Scheduled retraining jobs
│   │   ├── training_data_service.py       # Training data management
│   │   ├── disease_mapping.py             # Disease label mappings
│   │   ├── plant_info.py                  # Plant metadata lookup
│   │   ├── image_utils.py                 # Image preprocessing helpers
│   │   ├── scan_limits.py                 # Free/Pro scan limit enforcement
│   │   ├── geolocation.py                 # IP-based India free tier detection
│   │   ├── supabase_storage.py            # Image upload to Supabase Storage
│   │   ├── quality_gate.py                # Model quality checks before deploy
│   │   ├── custom_losses.py               # Custom ML loss functions
│   │   └── tflite_compat.py               # TFLite compatibility layer
│   ├── scripts/
│   │   ├── seed_coupons.py                # Seed coupon codes into DB
│   │   └── grant_pro_access.py            # Manually grant Pro to a user
│   └── main.py                            # FastAPI app entry point
├── requirements.txt                       # Pip dependencies
├── runtime.txt                            # Python version for deployment
├── render.yaml                            # Render.com deployment config
└── README.md
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/logout` - User logout

### Users
- `GET /api/v1/users/me` - Get current user profile
- `PUT /api/v1/users/me` - Update user profile
- `DELETE /api/v1/users/me` - Delete user account

### Plant Identification
- `POST /api/v1/identify/` - Identify plant from image
- `GET /api/v1/identify/history` - Get identification history

### Plant Diagnosis
- `POST /api/v1/diagnose/` - Diagnose plant health issues
- `GET /api/v1/diagnose/history` - Get diagnosis history

### Scans
- `POST /api/v1/scans/` - Upload scan
- `GET /api/v1/scans/` - Get user scans
- `GET /api/v1/scans/{scan_id}` - Get specific scan
- `DELETE /api/v1/scans/{scan_id}` - Delete scan

## Development

Run with hot reload:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```