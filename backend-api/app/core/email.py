"""Email service for team invitations and notifications.

In production, this would integrate with actual email providers like SendGrid,
Amazon SES, or similar services. For demo purposes, we'll log email content.
"""

import logging

logger = logging.getLogger("chimera.email")


async def send_invitation_email(
    email: str,
    workspace_name: str,
    invited_by: str,
    invitation_id: str,
    role: str,
    message: str | None = None,
) -> bool | None:
    """Send invitation email to user.

    In production, this would send actual emails via email service provider.
    For demo purposes, we log the email content.
    """
    try:
        # In production, would construct email template and send via provider
        email_content = f"""
        Subject: Invitation to join {workspace_name} on Chimera

        Hi,

        {invited_by} has invited you to join the '{workspace_name}' team workspace on Chimera
        as a {role}.

        {f"Message: {message}" if message else ""}

        To accept this invitation, click the link below:
        https://chimera.example.com/invite/{invitation_id}

        This invitation will expire in 7 days.

        Best regards,
        The Chimera Team
        """

        # In demo mode, just log the email
        logger.info(f"Email invitation sent to {email}: {email_content}")

        return True

    except Exception as e:
        logger.exception(f"Failed to send invitation email to {email}: {e}")
        return False


async def send_workspace_notification(
    email: str, workspace_name: str, subject: str, message: str
) -> bool | None:
    """Send workspace notification email."""
    try:
        email_content = f"""
        Subject: {subject}

        Hi,

        {message}

        Workspace: {workspace_name}

        Best regards,
        The Chimera Team
        """

        # In demo mode, just log the email
        logger.info(f"Notification email sent to {email}: {email_content}")

        return True

    except Exception as e:
        logger.exception(f"Failed to send notification email to {email}: {e}")
        return False
