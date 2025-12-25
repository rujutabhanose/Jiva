# app/services/email_service.py
import smtplib
from email.message import EmailMessage
from app.core.config import settings

SMTP_HOST = settings.SMTP_HOST
SMTP_PORT = settings.SMTP_PORT
SMTP_USER = settings.SMTP_USER
SMTP_PASS = settings.SMTP_PASS
FROM_EMAIL = settings.FROM_EMAIL or settings.SMTP_USER
FRONTEND_URL = settings.FRONTEND_URL

def send_verification_email(to_email: str, token: str):
    print(f"[EMAIL] Sending verification email to: {to_email}")
    print(f"[EMAIL] From: {FROM_EMAIL}")

    # Deep link for opening the app
    deep_link = f"jivaplants://verify-email?token={token}"

    # HTTP link for browser testing (emulator/web/physical device)
    # Use 192.168.0.121 for physical device testing on local network
    http_link = f"http://192.168.0.121:8000/api/v1/auth/verify-email?token={token}"

    # Use deep link as primary, HTTP link for testing
    verify_link = deep_link

    print(f"[EMAIL] Deep link: {deep_link}")
    print(f"[EMAIL] HTTP link (for testing): {http_link}")

    msg = EmailMessage()
    msg["Subject"] = "Verify your Jiva Plants account"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email

    # Plain text version
    msg.set_content(
        f"""Hi,

Thank you for creating a Jiva Plants account!

Click the link below to verify your email and complete your registration:

{verify_link}

This link will open the Jiva Plants app and automatically verify your account.

This link expires in 30 minutes.

If you didn't create this account, you can safely ignore this email.
"""
    )

    # HTML version with clickable link
    msg.add_alternative(
        f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
              <h2 style="color: #4CAF50;">Welcome to Jiva Plants!</h2>

              <p>Thank you for creating a Jiva Plants account!</p>

              <p>Click the button below to verify your email and complete your registration:</p>

              <div style="text-align: center; margin: 30px 0;">
                <a href="{verify_link}"
                   style="background-color: #4CAF50;
                          color: white;
                          padding: 14px 28px;
                          text-decoration: none;
                          border-radius: 8px;
                          display: inline-block;
                          font-weight: bold;">
                  Verify Email Address
                </a>
              </div>

              <p style="color: #666; font-size: 14px;">
                This link will open the Jiva Plants app and automatically verify your account.
              </p>

              <div style="background-color: #f5f5f5; padding: 16px; border-radius: 8px; margin: 20px 0;">
                <p style="color: #666; font-size: 13px; margin: 0 0 8px 0; font-weight: bold;">
                  🧪 Testing on emulator?
                </p>
                <p style="color: #666; font-size: 13px; margin: 0 0 8px 0;">
                  Copy and paste this link into your browser:
                </p>
                <p style="margin: 0;">
                  <a href="{http_link}" style="color: #4CAF50; word-break: break-all; font-size: 12px;">{http_link}</a>
                </p>
              </div>

              <p style="color: #999; font-size: 12px; margin-top: 30px;">
                This link expires in 30 minutes.<br>
                If you didn't create this account, you can safely ignore this email.
              </p>
            </div>
          </body>
        </html>
        """,
        subtype='html'
    )

    print(f"[EMAIL] Message To field: {msg['To']}")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

    print(f"[EMAIL] Verification email sent successfully to {to_email}")