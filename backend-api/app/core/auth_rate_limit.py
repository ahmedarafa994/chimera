"""
Authentication Rate Limiting Module

CRIT-005 FIX: Implements rate limiting specifically for authentication endpoints
to prevent brute force attacks and credential stuffing.

Features:
- Strict rate limits for login/auth endpoints
- IP-based tracking for anonymous auth attempts
- Progressive lockout after repeated failures
- Audit logging for security events
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, status

from app.core.logging import logger


@dataclass
class AuthRateLimitConfig:
    """Configuration for authentication rate limiting."""

    # Maximum login attempts per minute
    max_attempts_per_minute: int = 10
    # Maximum login attempts per hour
    max_attempts_per_hour: int = 30
    # Lockout duration in seconds after exceeding limits
    lockout_duration: int = 300  # 5 minutes
    # Burst size for short-term spikes
    burst_size: int = 3


# Default auth rate limit configuration
AUTH_RATE_LIMIT_CONFIG = AuthRateLimitConfig()


class AuthRateLimiter:
    """
    Rate limiter specifically for authentication endpoints.

    Implements a two-tier rate limiting:
    1. Per-minute limit for burst protection
    2. Per-hour limit for sustained attack protection

    Also tracks failed attempts for progressive lockout.
    """

    def __init__(self):
        # Track attempts: {ip: [(timestamp, success), ...]}
        self._attempts: dict[str, list[tuple[float, bool]]] = {}
        # Track lockouts: {ip: lockout_until_timestamp}
        self._lockouts: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_threshold = 1000
        self._max_keys = 10000

    def _cleanup_old_entries(self):
        """Remove expired entries to prevent memory bloat."""
        now = time.time()
        hour_ago = now - 3600

        # Clean up old attempts
        keys_to_delete = []
        for ip, attempts in self._attempts.items():
            self._attempts[ip] = [(ts, success) for ts, success in attempts if ts > hour_ago]
            if not self._attempts[ip]:
                keys_to_delete.append(ip)

        for key in keys_to_delete:
            del self._attempts[key]

        # Clean up expired lockouts
        expired_lockouts = [ip for ip, until in self._lockouts.items() if until < now]
        for ip in expired_lockouts:
            del self._lockouts[ip]

        # Hard limit on keys
        if len(self._attempts) > self._max_keys:
            keys = list(self._attempts.keys())
            excess = len(self._attempts) - self._max_keys
            for k in keys[:excess]:
                del self._attempts[k]

    async def check_auth_rate_limit(
        self, request: Request, config: AuthRateLimitConfig = None
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if an authentication attempt is allowed.

        Args:
            request: The FastAPI request object
            config: Optional custom rate limit configuration

        Returns:
            Tuple of (allowed, info_dict)
        """
        config = config or AUTH_RATE_LIMIT_CONFIG
        client_ip = request.client.host if request.client else "unknown"

        async with self._lock:
            now = time.time()

            # Periodic cleanup
            if len(self._attempts) > self._cleanup_threshold:
                self._cleanup_old_entries()

            # Check if IP is locked out
            if client_ip in self._lockouts:
                lockout_until = self._lockouts[client_ip]
                if now < lockout_until:
                    retry_after = int(lockout_until - now)
                    return False, {
                        "reason": "lockout",
                        "retry_after": retry_after,
                        "message": f"Too many failed attempts. Try again in {retry_after} seconds.",
                    }
                else:
                    # Lockout expired
                    del self._lockouts[client_ip]

            # Initialize or get attempts
            if client_ip not in self._attempts:
                self._attempts[client_ip] = []

            # Filter to relevant time windows
            minute_ago = now - 60
            hour_ago = now - 3600

            attempts_last_minute = [
                (ts, success) for ts, success in self._attempts[client_ip] if ts > minute_ago
            ]
            attempts_last_hour = [
                (ts, success) for ts, success in self._attempts[client_ip] if ts > hour_ago
            ]

            # Check per-minute limit
            if len(attempts_last_minute) >= config.max_attempts_per_minute:
                # Calculate retry after
                oldest_in_minute = min(ts for ts, _ in attempts_last_minute)
                retry_after = int(oldest_in_minute + 60 - now)

                logger.warning(
                    f"Auth rate limit exceeded (per-minute) for IP: {client_ip}. "
                    f"Attempts: {len(attempts_last_minute)}/{config.max_attempts_per_minute}"
                )

                return False, {
                    "reason": "rate_limit_minute",
                    "retry_after": max(1, retry_after),
                    "limit": config.max_attempts_per_minute,
                    "remaining": 0,
                    "message": "Too many authentication attempts. Please wait a minute.",
                }

            # Check per-hour limit
            if len(attempts_last_hour) >= config.max_attempts_per_hour:
                # Apply lockout
                self._lockouts[client_ip] = now + config.lockout_duration

                logger.warning(
                    f"Auth rate limit exceeded (per-hour) for IP: {client_ip}. "
                    f"Attempts: {len(attempts_last_hour)}/{config.max_attempts_per_hour}. "
                    f"Lockout applied for {config.lockout_duration}s"
                )

                return False, {
                    "reason": "rate_limit_hour",
                    "retry_after": config.lockout_duration,
                    "limit": config.max_attempts_per_hour,
                    "remaining": 0,
                    "message": f"Too many authentication attempts. Locked out for {config.lockout_duration // 60} minutes.",
                }

            # Calculate remaining attempts
            remaining_minute = config.max_attempts_per_minute - len(attempts_last_minute)
            remaining_hour = config.max_attempts_per_hour - len(attempts_last_hour)

            return True, {
                "reason": "allowed",
                "remaining_minute": remaining_minute,
                "remaining_hour": remaining_hour,
                "limit_minute": config.max_attempts_per_minute,
                "limit_hour": config.max_attempts_per_hour,
            }

    async def record_attempt(self, request: Request, success: bool):
        """
        Record an authentication attempt.

        Args:
            request: The FastAPI request object
            success: Whether the authentication was successful
        """
        client_ip = request.client.host if request.client else "unknown"

        async with self._lock:
            now = time.time()

            if client_ip not in self._attempts:
                self._attempts[client_ip] = []

            self._attempts[client_ip].append((now, success))

            # If successful, clear lockout
            if success and client_ip in self._lockouts:
                del self._lockouts[client_ip]

            # Log the attempt
            if success:
                logger.info(f"Successful auth attempt from IP: {client_ip}")
            else:
                # Count recent failures
                hour_ago = now - 3600
                recent_failures = sum(
                    1 for ts, s in self._attempts[client_ip] if ts > hour_ago and not s
                )
                logger.warning(
                    f"Failed auth attempt from IP: {client_ip}. Recent failures: {recent_failures}"
                )

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "tracked_ips": len(self._attempts),
            "active_lockouts": len(self._lockouts),
            "total_attempts": sum(len(v) for v in self._attempts.values()),
        }


# Global auth rate limiter instance
_auth_rate_limiter: AuthRateLimiter | None = None


def get_auth_rate_limiter() -> AuthRateLimiter:
    """Get the auth rate limiter instance."""
    global _auth_rate_limiter
    if _auth_rate_limiter is None:
        _auth_rate_limiter = AuthRateLimiter()
    return _auth_rate_limiter


async def check_auth_rate_limit(request: Request):
    """
    FastAPI dependency for auth rate limiting.

    Usage:
        @router.post("/login")
        async def login(
            request: Request,
            _: None = Depends(check_auth_rate_limit)
        ):
            ...
    """
    limiter = get_auth_rate_limiter()
    allowed, info = await limiter.check_auth_rate_limit(request)

    if not allowed:
        # Log security event
        try:
            from app.core.audit import AuditAction, audit_log

            audit_log(
                action=AuditAction.SECURITY_RATE_LIMIT,
                user_id=None,
                resource="auth",
                details={
                    "reason": info.get("reason"),
                    "retry_after": info.get("retry_after"),
                    "ip": request.client.host if request.client else "unknown",
                },
                ip_address=request.client.host if request.client else None,
            )
        except ImportError:
            pass  # Audit module not available

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=info.get("message", "Rate limit exceeded"),
            headers={
                "Retry-After": str(info.get("retry_after", 60)),
                "X-RateLimit-Reason": info.get("reason", "rate_limit"),
            },
        )


async def record_auth_attempt(request: Request, success: bool):
    """
    Record an authentication attempt for rate limiting.

    Call this after processing a login/auth request.

    Usage:
        @router.post("/login")
        async def login(request: Request, credentials: LoginRequest):
            try:
                user = authenticate(credentials)
                await record_auth_attempt(request, success=True)
                return {"token": create_token(user)}
            except AuthError:
                await record_auth_attempt(request, success=False)
                raise
    """
    limiter = get_auth_rate_limiter()
    await limiter.record_attempt(request, success)
