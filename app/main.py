from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import warnings
from app.api.v1.router import api_router
from app.core.config import settings
from app.db.init_db import init_db
from app.db.session import SessionLocal
from app.services.image_utils import cleanup_old_images, UPLOAD_DIR
from app.services.model_manager import model_manager
from app.services.model_scheduler import init_scheduler, shutdown_scheduler

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

    # Startup: Initialize Model Manager for continuous learning
    print("Loading model versions...")
    try:
        db = SessionLocal()
        model_manager.initialize(db)
        db.close()
        print("Model manager initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize model manager: {e}")

    # Startup: Initialize training scheduler (weekly retraining)
    print("Initializing training scheduler...")
    try:
        init_scheduler()
        print("Training scheduler initialized (weekly on Sundays at 2:00 AM UTC)")
    except Exception as e:
        print(f"Warning: Failed to initialize scheduler: {e}")

    yield

    # Shutdown: Stop training scheduler
    print("Shutting down training scheduler...")
    shutdown_scheduler()

    # Shutdown: Cancel background tasks
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Jiva Plants API - Backend for plant care management",
    lifespan=lifespan
)

# CORS middleware configuration for mobile app support
# Always use specific origins with credentials enabled to support JWT authentication
# Cannot use allow_origins=["*"] with allow_credentials=True (CORS spec violation)
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


@app.get("/delete-account", response_class=HTMLResponse)
async def delete_account_page():
    """Web page for account deletion (required for App Store compliance)"""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Delete Account – Jiva Plants</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f5;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      padding: 40px 32px;
      max-width: 420px;
      width: 100%;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .logo { font-size: 32px; text-align: center; margin-bottom: 8px; }
    h1 { font-size: 22px; font-weight: 700; text-align: center; color: #111; margin-bottom: 6px; }
    .subtitle { font-size: 14px; color: #666; text-align: center; margin-bottom: 28px; }
    label { display: block; font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; }
    input {
      width: 100%; padding: 12px 14px; border: 1.5px solid #ddd;
      border-radius: 10px; font-size: 15px; outline: none;
      transition: border-color .2s;
    }
    input:focus { border-color: #e74c3c; }
    .field { margin-bottom: 18px; }
    .btn {
      width: 100%; padding: 14px; border: none; border-radius: 10px;
      font-size: 16px; font-weight: 600; cursor: pointer; transition: opacity .2s;
    }
    .btn:disabled { opacity: .5; cursor: not-allowed; }
    .btn-primary { background: #e74c3c; color: #fff; }
    .btn-primary:hover:not(:disabled) { opacity: .88; }
    .btn-secondary {
      background: #f0f0f0; color: #333; margin-top: 10px;
    }
    .btn-secondary:hover:not(:disabled) { background: #e4e4e4; }
    .warning {
      background: #fff5f5; border: 1.5px solid #fca5a5;
      border-radius: 10px; padding: 14px; margin-bottom: 22px; font-size: 13px; color: #b91c1c;
    }
    .warning ul { padding-left: 18px; margin-top: 6px; }
    .warning li { margin-top: 4px; }
    .msg {
      margin-top: 18px; padding: 12px 14px; border-radius: 10px;
      font-size: 14px; text-align: center; display: none;
    }
    .msg.error { background: #fff5f5; color: #b91c1c; display: block; }
    .msg.success { background: #f0fdf4; color: #15803d; display: block; }
    #step2 { display: none; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🌿</div>
    <h1>Delete Account</h1>
    <p class="subtitle">Jiva Plants</p>

    <!-- Step 1: Login -->
    <div id="step1">
      <div class="warning">
        <strong>Warning:</strong> Deleting your account is permanent and cannot be undone. This will remove:
        <ul>
          <li>Your profile and personal data</li>
          <li>All scan history</li>
          <li>Your plant care records</li>
          <li>Any active subscription</li>
        </ul>
      </div>
      <div class="field">
        <label for="email">Email</label>
        <input type="email" id="email" placeholder="you@example.com" autocomplete="email" />
      </div>
      <div class="field">
        <label for="password">Password</label>
        <input type="password" id="password" placeholder="••••••••" autocomplete="current-password" />
      </div>
      <button class="btn btn-primary" id="loginBtn" onclick="handleLogin()">Continue</button>
      <div id="msg1" class="msg"></div>
    </div>

    <!-- Step 2: Confirm deletion -->
    <div id="step2">
      <div class="warning">
        <strong>Are you absolutely sure?</strong><br/><br/>
        You are about to permanently delete your account <strong id="userEmail"></strong>.<br/><br/>
        This action <strong>cannot be undone</strong>.
      </div>
      <button class="btn btn-primary" id="deleteBtn" onclick="handleDelete()">Yes, Delete My Account</button>
      <button class="btn btn-secondary" onclick="goBack()">Cancel</button>
      <div id="msg2" class="msg"></div>
    </div>
  </div>

  <script>
    let authToken = null;

    function showMsg(id, text, type) {
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = 'msg ' + type;
    }

    async function handleLogin() {
      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value;
      const btn = document.getElementById('loginBtn');

      if (!email || !password) {
        showMsg('msg1', 'Please enter your email and password.', 'error');
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Signing in…';

      try {
        const res = await fetch('/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        if (!res.ok) {
          showMsg('msg1', data.detail || 'Invalid email or password.', 'error');
          return;
        }
        authToken = data.access_token;
        document.getElementById('userEmail').textContent = email;
        document.getElementById('step1').style.display = 'none';
        document.getElementById('step2').style.display = 'block';
      } catch (e) {
        showMsg('msg1', 'Network error. Please try again.', 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Continue';
      }
    }

    async function handleDelete() {
      const btn = document.getElementById('deleteBtn');
      btn.disabled = true;
      btn.textContent = 'Deleting…';

      try {
        const res = await fetch('/api/v1/users/me', {
          method: 'DELETE',
          headers: { 'Authorization': 'Bearer ' + authToken }
        });
        if (!res.ok) {
          const data = await res.json();
          showMsg('msg2', data.detail || 'Failed to delete account. Please try again.', 'error');
          btn.disabled = false;
          btn.textContent = 'Yes, Delete My Account';
          return;
        }
        document.getElementById('deleteBtn').style.display = 'none';
        document.querySelector('.btn-secondary').style.display = 'none';
        showMsg('msg2', 'Your account has been permanently deleted. Goodbye!', 'success');
      } catch (e) {
        showMsg('msg2', 'Network error. Please try again.', 'error');
        btn.disabled = false;
        btn.textContent = 'Yes, Delete My Account';
      }
    }

    function goBack() {
      document.getElementById('step2').style.display = 'none';
      document.getElementById('step1').style.display = 'block';
      authToken = null;
    }

    document.getElementById('password').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') handleLogin();
    });
  </script>
</body>
</html>""")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )