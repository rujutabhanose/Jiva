# app/core/email.py
import resend
from app.core.config import settings


def init_resend():
    """Initialize Resend with API key"""
    if settings.RESEND_API_KEY:
        resend.api_key = settings.RESEND_API_KEY


def send_password_reset_otp(email: str, otp: str, name: str = None) -> bool:
    """
    Send password reset OTP via email.

    Args:
        email: Recipient email address
        otp: 6-digit OTP code
        name: Optional user name for personalization

    Returns:
        True if email sent successfully, False otherwise
    """
    if not settings.RESEND_API_KEY:
        print(f"[DEV MODE] Password reset OTP for {email}: {otp}")
        return True

    init_resend()

    greeting = f"Hi {name}," if name else "Hi,"

    try:
        print(f"[Email] Sending password reset OTP to {email} from {settings.EMAIL_FROM}")
        result = resend.Emails.send({
            "from": settings.EMAIL_FROM,
            "to": email,
            "subject": "Reset Your Jiva Password",
            "html": f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #2d5a27;">Jiva Plants</h2>
                <p>{greeting}</p>
                <p>You requested to reset your password. Use the code below to continue:</p>
                <div style="background-color: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 8px;">
                    <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #2d5a27;">{otp}</span>
                </div>
                <p>This code expires in <strong>15 minutes</strong>.</p>
                <p>If you didn't request this, you can safely ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="color: #888; font-size: 12px;">Jiva Plants - Your plant care companion</p>
            </div>
            """
        })
        print(f"[Email] Successfully sent! Response: {result}")
        return True
    except Exception as e:
        print(f"[Email] Failed to send password reset email: {e}")
        print(f"[Email] Error type: {type(e).__name__}")
        return False
