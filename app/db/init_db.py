from app.db.session import engine
from app.models import user, scan, coupon, coupon_redemption, diagnosis_feedback
from app.models import training_data, model_version  # Continuous learning models
from app.models import password_reset  # Password reset tokens
from app.db.base import Base

def init_db():
    Base.metadata.create_all(bind=engine)