"""Email Service - SMTP-based email delivery with template rendering.

Provides:
- Async SMTP email delivery
- HTML and plain text email templates
- Template rendering with Jinja2
- Graceful fallback for development (log instead of send)
- Configurable via environment variables

Usage:
    from app.services.email_service import email_service

    # Send verification email
    await email_service.send_verification_email(
        email="user@example.com",
        username="john",
        verification_url="https://app.example.com/verify?token=abc123"
    )

    # Send password reset email
    await email_service.send_password_reset_email(
        email="user@example.com",
        username="john",
        reset_url="https://app.example.com/reset?token=abc123"
    )
"""

import asyncio
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Email Configuration
# =============================================================================


class SMTPSecurityMode(str, Enum):
    """SMTP security modes."""

    NONE = "none"  # Plain SMTP (port 25)
    STARTTLS = "starttls"  # STARTTLS (port 587)
    SSL_TLS = "ssl_tls"  # SSL/TLS (port 465)


@dataclass
class EmailConfig:
    """Email service configuration from environment variables.

    Environment Variables:
        SMTP_HOST: SMTP server hostname (default: localhost)
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USERNAME: SMTP authentication username
        SMTP_PASSWORD: SMTP authentication password
        SMTP_SECURITY: Security mode (none, starttls, ssl_tls)
        SMTP_FROM_EMAIL: Default sender email address
        SMTP_FROM_NAME: Default sender display name
        EMAIL_DEV_MODE: If true, log emails instead of sending (default: true in dev)
        EMAIL_VERIFICATION_URL: Base URL for email verification
        EMAIL_PASSWORD_RESET_URL: Base URL for password reset
        EMAIL_FRONTEND_URL: Frontend application URL
    """

    host: str
    port: int
    username: str | None
    password: str | None
    security: SMTPSecurityMode
    from_email: str
    from_name: str
    dev_mode: bool
    verification_url: str
    password_reset_url: str
    frontend_url: str
    timeout: int

    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Load configuration from environment variables."""
        environment = os.getenv("ENVIRONMENT", "development")
        default_dev_mode = environment in ("development", "local", "test")

        return cls(
            host=os.getenv("SMTP_HOST", "localhost"),
            port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USERNAME"),
            password=os.getenv("SMTP_PASSWORD"),
            security=SMTPSecurityMode(os.getenv("SMTP_SECURITY", "starttls").lower()),
            from_email=os.getenv("SMTP_FROM_EMAIL", "noreply@chimera.local"),
            from_name=os.getenv("SMTP_FROM_NAME", "Chimera"),
            dev_mode=os.getenv("EMAIL_DEV_MODE", str(default_dev_mode)).lower()
            in ("true", "1", "yes"),
            verification_url=os.getenv(
                "EMAIL_VERIFICATION_URL",
                "http://localhost:3001/verify-email",
            ),
            password_reset_url=os.getenv(
                "EMAIL_PASSWORD_RESET_URL",
                "http://localhost:3001/reset-password",
            ),
            frontend_url=os.getenv(
                "EMAIL_FRONTEND_URL",
                "http://localhost:3001",
            ),
            timeout=int(os.getenv("SMTP_TIMEOUT", "30")),
        )


# =============================================================================
# Email Templates (Inline for Simplicity)
# =============================================================================


class EmailTemplates:
    """Email templates with HTML and plain text versions."""

    # Common styles for HTML emails
    COMMON_STYLES = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 600px;
            margin: 20px auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }
        .content {
            padding: 40px 30px;
        }
        .button {
            display: inline-block;
            padding: 14px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .button:hover {
            opacity: 0.9;
        }
        .footer {
            padding: 20px 30px;
            background: #f8f9fa;
            text-align: center;
            color: #6c757d;
            font-size: 13px;
        }
        .footer a {
            color: #667eea;
            text-decoration: none;
        }
        .code-box {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            font-family: monospace;
            font-size: 14px;
            word-break: break-all;
            margin: 15px 0;
        }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 16px;
            margin: 15px 0;
            color: #856404;
        }
    """

    @staticmethod
    def verification_html(
        username: str,
        verification_url: str,
        frontend_url: str,
    ) -> str:
        """Generate verification email HTML."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verify Your Email - Chimera</title>
    <style>{EmailTemplates.COMMON_STYLES}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to Chimera</h1>
        </div>
        <div class="content">
            <h2>Hi {username}!</h2>
            <p>Thanks for signing up for Chimera. To complete your registration and access all features, please verify your email address.</p>

            <p style="text-align: center;">
                <a href="{verification_url}" class="button">Verify Email Address</a>
            </p>

            <p>If the button doesn't work, copy and paste this link into your browser:</p>
            <div class="code-box">{verification_url}</div>

            <div class="warning">
                <strong>Note:</strong> This link will expire in 24 hours. If you didn't create an account with Chimera, please ignore this email.
            </div>
        </div>
        <div class="footer">
            <p>This email was sent by <a href="{frontend_url}">Chimera</a></p>
            <p>&copy; 2026 Chimera. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def verification_text(
        username: str,
        verification_url: str,
    ) -> str:
        """Generate verification email plain text."""
        return f"""
Welcome to Chimera!

Hi {username},

Thanks for signing up for Chimera. To complete your registration and access all features, please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

If you didn't create an account with Chimera, please ignore this email.

---
Chimera - AI-Powered Prompt Engineering Platform
"""

    @staticmethod
    def password_reset_html(
        username: str,
        reset_url: str,
        frontend_url: str,
    ) -> str:
        """Generate password reset email HTML."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Your Password - Chimera</title>
    <style>{EmailTemplates.COMMON_STYLES}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset</h1>
        </div>
        <div class="content">
            <h2>Hi {username}!</h2>
            <p>We received a request to reset your password for your Chimera account. Click the button below to create a new password:</p>

            <p style="text-align: center;">
                <a href="{reset_url}" class="button">Reset Password</a>
            </p>

            <p>If the button doesn't work, copy and paste this link into your browser:</p>
            <div class="code-box">{reset_url}</div>

            <div class="warning">
                <strong>Security Notice:</strong> This link will expire in 1 hour. If you didn't request a password reset, please ignore this email or contact support if you're concerned about your account security.
            </div>
        </div>
        <div class="footer">
            <p>This email was sent by <a href="{frontend_url}">Chimera</a></p>
            <p>&copy; 2026 Chimera. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def password_reset_text(
        username: str,
        reset_url: str,
    ) -> str:
        """Generate password reset email plain text."""
        return f"""
Password Reset Request

Hi {username},

We received a request to reset your password for your Chimera account.

To reset your password, click the link below:
{reset_url}

This link will expire in 1 hour.

If you didn't request a password reset, please ignore this email. Your password will remain unchanged.

---
Chimera - AI-Powered Prompt Engineering Platform
"""

    @staticmethod
    def welcome_html(
        username: str,
        frontend_url: str,
    ) -> str:
        """Generate welcome email HTML (sent after verification)."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Chimera!</title>
    <style>{EmailTemplates.COMMON_STYLES}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>You're All Set!</h1>
        </div>
        <div class="content">
            <h2>Welcome, {username}!</h2>
            <p>Your email has been verified and your Chimera account is now fully activated. You're ready to explore the power of AI-driven prompt engineering and security research.</p>

            <h3>Getting Started:</h3>
            <ul style="padding-left: 20px;">
                <li><strong>Dashboard</strong> - View your prompts and campaigns</li>
                <li><strong>Prompt Generator</strong> - Create and enhance prompts with AI</li>
                <li><strong>Jailbreak Research</strong> - Explore advanced prompt techniques</li>
                <li><strong>API Access</strong> - Integrate Chimera into your workflow</li>
            </ul>

            <p style="text-align: center;">
                <a href="{frontend_url}/dashboard" class="button">Go to Dashboard</a>
            </p>

            <p>If you have any questions, check out our documentation or reach out to our support team.</p>
        </div>
        <div class="footer">
            <p>Thanks for joining <a href="{frontend_url}">Chimera</a>!</p>
            <p>&copy; 2026 Chimera. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def welcome_text(
        username: str,
        frontend_url: str,
    ) -> str:
        """Generate welcome email plain text."""
        return f"""
Welcome to Chimera, {username}!

Your email has been verified and your account is now fully activated.

Getting Started:
- Dashboard: View your prompts and campaigns
- Prompt Generator: Create and enhance prompts with AI
- Jailbreak Research: Explore advanced prompt techniques
- API Access: Integrate Chimera into your workflow

Visit your dashboard: {frontend_url}/dashboard

If you have any questions, check out our documentation or reach out to support.

---
Chimera - AI-Powered Prompt Engineering Platform
"""

    @staticmethod
    def invitation_html(
        inviter_name: str,
        role: str,
        invitation_url: str,
        frontend_url: str,
    ) -> str:
        """Generate invitation email HTML (for admin invites)."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>You're Invited to Chimera</title>
    <style>{EmailTemplates.COMMON_STYLES}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>You're Invited!</h1>
        </div>
        <div class="content">
            <h2>Join Chimera</h2>
            <p><strong>{inviter_name}</strong> has invited you to join Chimera as a <strong>{role}</strong>.</p>

            <p>Chimera is an AI-powered platform for prompt engineering and security research. Click below to accept the invitation and set up your account:</p>

            <p style="text-align: center;">
                <a href="{invitation_url}" class="button">Accept Invitation</a>
            </p>

            <p>If the button doesn't work, copy and paste this link into your browser:</p>
            <div class="code-box">{invitation_url}</div>

            <div class="warning">
                <strong>Note:</strong> This invitation will expire in 7 days. If you don't know {inviter_name} or weren't expecting this invitation, please ignore this email.
            </div>
        </div>
        <div class="footer">
            <p>This email was sent by <a href="{frontend_url}">Chimera</a></p>
            <p>&copy; 2026 Chimera. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

    @staticmethod
    def invitation_text(
        inviter_name: str,
        role: str,
        invitation_url: str,
    ) -> str:
        """Generate invitation email plain text."""
        return f"""
You're Invited to Join Chimera!

{inviter_name} has invited you to join Chimera as a {role}.

Chimera is an AI-powered platform for prompt engineering and security research.

To accept the invitation and set up your account, click the link below:
{invitation_url}

This invitation will expire in 7 days.

If you don't know {inviter_name} or weren't expecting this invitation, please ignore this email.

---
Chimera - AI-Powered Prompt Engineering Platform
"""


# =============================================================================
# Email Result
# =============================================================================


@dataclass
class EmailResult:
    """Result of email send operation."""

    success: bool
    message_id: str | None = None
    error: str | None = None
    dev_mode: bool = False


# =============================================================================
# Email Service
# =============================================================================


class EmailService:
    """Async email service with SMTP support.

    Features:
    - HTML and plain text email support
    - Template rendering for common email types
    - Configurable SMTP with SSL/TLS/STARTTLS
    - Development mode (log instead of send)
    - Graceful error handling
    """

    def __init__(self, config: EmailConfig | None = None) -> None:
        """Initialize email service.

        Args:
            config: Email configuration (loads from env if not provided)

        """
        self._config = config or EmailConfig.from_env()

    @property
    def config(self) -> EmailConfig:
        """Get current email configuration."""
        return self._config

    @property
    def is_dev_mode(self) -> bool:
        """Check if running in development mode."""
        return self._config.dev_mode

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
        reply_to: str | None = None,
    ) -> EmailResult:
        """Send an email asynchronously.

        In development mode, logs the email instead of sending.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            text_content: Plain text email body (optional)
            from_email: Sender email (uses default if not provided)
            from_name: Sender name (uses default if not provided)
            reply_to: Reply-to address (optional)

        Returns:
            EmailResult with send status

        """
        sender_email = from_email or self._config.from_email
        sender_name = from_name or self._config.from_name

        # In dev mode, log instead of sending
        if self._config.dev_mode:
            return await self._log_email(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                sender_email=sender_email,
                sender_name=sender_name,
            )

        # Build the email message
        message = self._build_message(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            sender_email=sender_email,
            sender_name=sender_name,
            reply_to=reply_to,
        )

        # Send email in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_smtp,
            message,
            to_email,
        )

    def _build_message(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str | None,
        sender_email: str,
        sender_name: str,
        reply_to: str | None,
    ) -> MIMEMultipart:
        """Build MIME message for email."""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{sender_name} <{sender_email}>"
        message["To"] = to_email

        if reply_to:
            message["Reply-To"] = reply_to

        # Add plain text version first (fallback)
        if text_content:
            message.attach(MIMEText(text_content, "plain", "utf-8"))

        # Add HTML version (preferred)
        message.attach(MIMEText(html_content, "html", "utf-8"))

        return message

    def _send_smtp(
        self,
        message: MIMEMultipart,
        to_email: str,
    ) -> EmailResult:
        """Send email via SMTP (runs in thread pool)."""
        try:
            if self._config.security == SMTPSecurityMode.SSL_TLS:
                # SSL/TLS connection (port 465)
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self._config.host,
                    self._config.port,
                    context=context,
                    timeout=self._config.timeout,
                ) as server:
                    return self._authenticate_and_send(server, message, to_email)

            else:
                # Plain or STARTTLS connection
                with smtplib.SMTP(
                    self._config.host,
                    self._config.port,
                    timeout=self._config.timeout,
                ) as server:
                    if self._config.security == SMTPSecurityMode.STARTTLS:
                        context = ssl.create_default_context()
                        server.starttls(context=context)

                    return self._authenticate_and_send(server, message, to_email)

        except smtplib.SMTPAuthenticationError as e:
            logger.exception(f"SMTP authentication failed: {e}")
            return EmailResult(
                success=False,
                error="SMTP authentication failed. Check username and password.",
            )

        except smtplib.SMTPException as e:
            logger.exception(f"SMTP error sending email to {to_email}: {e}")
            return EmailResult(
                success=False,
                error=f"SMTP error: {e}",
            )

        except TimeoutError:
            logger.exception(f"SMTP timeout connecting to {self._config.host}")
            return EmailResult(
                success=False,
                error="Connection to email server timed out",
            )

        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}", exc_info=True)
            return EmailResult(
                success=False,
                error=f"Unexpected error: {e}",
            )

    def _authenticate_and_send(
        self,
        server: smtplib.SMTP,
        message: MIMEMultipart,
        to_email: str,
    ) -> EmailResult:
        """Authenticate and send email via SMTP server."""
        # Authenticate if credentials provided
        if self._config.username and self._config.password:
            server.login(self._config.username, self._config.password)

        # Send the email
        server.send_message(message)

        logger.info(f"Email sent successfully to {to_email}")
        return EmailResult(
            success=True,
            message_id=message.get("Message-ID"),
        )

    async def _log_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        sender_email: str,
        sender_name: str,
    ) -> EmailResult:
        """Log email in development mode instead of sending."""
        logger.info(
            f"\n{'=' * 60}\n"
            f"EMAIL (DEV MODE - NOT SENT)\n"
            f"{'=' * 60}\n"
            f"From: {sender_name} <{sender_email}>\n"
            f"To: {to_email}\n"
            f"Subject: {subject}\n"
            f"{'=' * 60}\n"
            f"(HTML content not displayed in logs)\n"
            f"{'=' * 60}\n",
        )
        return EmailResult(
            success=True,
            message_id=f"dev-{id(html_content)}",
            dev_mode=True,
        )

    # =========================================================================
    # Convenience Methods for Common Email Types
    # =========================================================================

    async def send_verification_email(
        self,
        email: str,
        username: str,
        verification_token: str,
    ) -> EmailResult:
        """Send email verification email.

        Args:
            email: Recipient email address
            username: User's display name
            verification_token: Verification token to include in URL

        Returns:
            EmailResult with send status

        """
        verification_url = f"{self._config.verification_url}?token={verification_token}"

        html_content = EmailTemplates.verification_html(
            username=username,
            verification_url=verification_url,
            frontend_url=self._config.frontend_url,
        )
        text_content = EmailTemplates.verification_text(
            username=username,
            verification_url=verification_url,
        )

        return await self.send_email(
            to_email=email,
            subject="Verify Your Email - Chimera",
            html_content=html_content,
            text_content=text_content,
        )

    async def send_password_reset_email(
        self,
        email: str,
        username: str,
        reset_token: str,
    ) -> EmailResult:
        """Send password reset email.

        Args:
            email: Recipient email address
            username: User's display name
            reset_token: Password reset token to include in URL

        Returns:
            EmailResult with send status

        """
        reset_url = f"{self._config.password_reset_url}?token={reset_token}"

        html_content = EmailTemplates.password_reset_html(
            username=username,
            reset_url=reset_url,
            frontend_url=self._config.frontend_url,
        )
        text_content = EmailTemplates.password_reset_text(
            username=username,
            reset_url=reset_url,
        )

        return await self.send_email(
            to_email=email,
            subject="Reset Your Password - Chimera",
            html_content=html_content,
            text_content=text_content,
        )

    async def send_welcome_email(
        self,
        email: str,
        username: str,
    ) -> EmailResult:
        """Send welcome email after email verification.

        Args:
            email: Recipient email address
            username: User's display name

        Returns:
            EmailResult with send status

        """
        html_content = EmailTemplates.welcome_html(
            username=username,
            frontend_url=self._config.frontend_url,
        )
        text_content = EmailTemplates.welcome_text(
            username=username,
            frontend_url=self._config.frontend_url,
        )

        return await self.send_email(
            to_email=email,
            subject="Welcome to Chimera!",
            html_content=html_content,
            text_content=text_content,
        )

    async def send_invitation_email(
        self,
        email: str,
        inviter_name: str,
        role: str,
        invitation_token: str,
    ) -> EmailResult:
        """Send invitation email (for admin invites).

        Args:
            email: Recipient email address
            inviter_name: Name of the person who sent the invitation
            role: Role being assigned (e.g., "Researcher", "Viewer")
            invitation_token: Invitation token to include in URL

        Returns:
            EmailResult with send status

        """
        # Invitation URL goes to a special invite acceptance page
        invitation_url = f"{self._config.frontend_url}/accept-invite?token={invitation_token}"

        html_content = EmailTemplates.invitation_html(
            inviter_name=inviter_name,
            role=role,
            invitation_url=invitation_url,
            frontend_url=self._config.frontend_url,
        )
        text_content = EmailTemplates.invitation_text(
            inviter_name=inviter_name,
            role=role,
            invitation_url=invitation_url,
        )

        return await self.send_email(
            to_email=email,
            subject=f"{inviter_name} invited you to Chimera",
            html_content=html_content,
            text_content=text_content,
        )


# =============================================================================
# Global Instance and Factory
# =============================================================================


# Global email service instance
_email_service: EmailService | None = None


def get_email_service() -> EmailService:
    """Get or create the global email service instance.

    Returns:
        EmailService instance

    """
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service


def create_email_service(config: EmailConfig | None = None) -> EmailService:
    """Create a new email service instance with optional config.

    Args:
        config: Optional email configuration

    Returns:
        New EmailService instance

    """
    return EmailService(config)


# Convenience alias for direct imports
email_service = get_email_service()
