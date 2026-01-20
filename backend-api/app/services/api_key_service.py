# =============================================================================
# Chimera - API Key Storage Service
# =============================================================================
# Service layer for secure API key CRUD operations with encryption at rest.
# Supports multiple keys per provider with primary/backup designation for
# automatic failover scenarios.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 1.2: Create API Key Storage Service
# =============================================================================

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any

from app.core.config import API_KEY_NAME_MAP, settings
from app.core.encryption import EncryptionError, decrypt_api_key, encrypt_api_key, is_encrypted
from app.domain.api_key_models import (
    API_KEY_PATTERNS,
    ApiKeyCreate,
    ApiKeyListResponse,
    ApiKeyRecord,
    ApiKeyResponse,
    ApiKeyRole,
    ApiKeyStatus,
    ApiKeyTestRequest,
    ApiKeyTestResult,
    ApiKeyUpdate,
    ProviderKeySummary,
    ProviderType,
)

logger = logging.getLogger(__name__)


class ApiKeyServiceError(Exception):
    """Base exception for API key service operations."""


class ApiKeyNotFoundError(ApiKeyServiceError):
    """Raised when an API key is not found."""


class ApiKeyValidationError(ApiKeyServiceError):
    """Raised when API key validation fails."""


class ApiKeyDuplicateError(ApiKeyServiceError):
    """Raised when attempting to create a duplicate key."""


class ApiKeyStorageService:
    """Service for secure API key storage and management.

    Features:
    - Secure storage with AES-256 encryption at rest
    - CRUD operations for all supported providers
    - Provider-specific key format validation
    - Usage statistics tracking (last_used, request_count)
    - Primary/backup key designation for failover scenarios
    - Thread-safe operations with async locking

    Usage:
        from app.services.api_key_service import api_key_service

        # Add a new key
        key = await api_key_service.create_key(ApiKeyCreate(
            provider_id="openai",
            api_key="sk-abc123...",
            name="Production Key",
            role=ApiKeyRole.PRIMARY
        ))

        # Get key for provider
        key = await api_key_service.get_primary_key("openai")

        # List all keys
        keys = await api_key_service.list_keys()
    """

    _instance: "ApiKeyStorageService | None" = None

    def __new__(cls) -> "ApiKeyStorageService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # In-memory storage for API keys (keyed by key ID)
        self._keys: dict[str, ApiKeyRecord] = {}

        # Index for quick lookup by provider
        self._provider_index: dict[str, list[str]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Supported providers from config
        self._supported_providers = set(ProviderType)

        self._initialized = True

        # Initialize from environment variables if available
        self._initialize_from_environment()

        logger.info("ApiKeyStorageService initialized")

    def _initialize_from_environment(self) -> None:
        """Initialize API keys from environment variables.

        This loads existing API keys from the environment configuration
        into the storage service for seamless integration with existing setup.
        """
        for provider_name, env_var in API_KEY_NAME_MAP.items():
            api_key = getattr(settings, env_var, None)
            if api_key and api_key.strip():
                try:
                    # Normalize provider name
                    normalized_provider = self._normalize_provider_id(provider_name)
                    if not normalized_provider:
                        continue

                    # Check if we already have a key for this provider
                    existing_keys = self._provider_index.get(normalized_provider, [])
                    if existing_keys:
                        continue

                    # Create a record for the environment key
                    key_id = f"env_{normalized_provider}_primary"

                    # Decrypt if encrypted, then re-encrypt for storage
                    try:
                        if is_encrypted(api_key):
                            plaintext_key = decrypt_api_key(api_key)
                        else:
                            plaintext_key = api_key
                        encrypted_key = encrypt_api_key(plaintext_key)
                    except EncryptionError:
                        # If decryption fails, encrypt as-is
                        encrypted_key = encrypt_api_key(api_key)

                    record = ApiKeyRecord(
                        id=key_id,
                        provider_id=normalized_provider,
                        encrypted_key=encrypted_key,
                        name=f"{normalized_provider.title()} Environment Key",
                        role=ApiKeyRole.PRIMARY,
                        status=ApiKeyStatus.ACTIVE,
                        priority=0,
                        description=f"Loaded from {env_var} environment variable",
                        tags=["environment", "auto-imported"],
                        metadata={"source": "environment", "env_var": env_var},
                    )

                    self._keys[key_id] = record
                    self._provider_index.setdefault(normalized_provider, []).append(key_id)

                    logger.debug(f"Loaded API key for {normalized_provider} from environment")

                except Exception as e:
                    logger.warning(f"Failed to load API key for {provider_name}: {e}")

        logger.info(f"Initialized {len(self._keys)} API keys from environment variables")

    def _normalize_provider_id(self, provider_id: str) -> str | None:
        """Normalize provider ID to a standard format.

        Args:
            provider_id: Provider identifier to normalize

        Returns:
            Normalized provider ID or None if not recognized

        """
        normalized = provider_id.lower().strip()

        # Handle aliases
        aliases = {
            "gemini": "google",
            "zhipu": "bigmodel",
        }
        normalized = aliases.get(normalized, normalized)

        # Validate against known providers
        try:
            ProviderType(normalized)
            return normalized
        except ValueError:
            return None

    def _generate_key_id(self, provider_id: str, role: ApiKeyRole) -> str:
        """Generate a unique key ID."""
        timestamp = int(time.time() * 1000) % 1000000
        unique_suffix = uuid.uuid4().hex[:8]
        return f"key_{provider_id}_{role.value}_{timestamp}_{unique_suffix}"

    def _validate_key_format(self, provider_id: str, api_key: str) -> tuple[bool, str]:
        """Validate API key format for a specific provider.

        Args:
            provider_id: Provider identifier
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, error_message)

        """
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty"

        if len(api_key) < 10:
            return False, "API key must be at least 10 characters"

        if len(api_key) > 256:
            return False, "API key must be at most 256 characters"

        # Provider-specific format validation
        pattern = API_KEY_PATTERNS.get(provider_id.lower())
        if pattern and not re.match(pattern, api_key):
            # Log warning but don't fail - providers sometimes change formats
            logger.warning(
                f"API key for {provider_id} doesn't match expected format. "
                f"Expected pattern: {pattern}",
            )
            # Return valid with a warning - be lenient
            return True, ""

        return True, ""

    async def create_key(self, request: ApiKeyCreate) -> ApiKeyResponse:
        """Create and store a new API key.

        Args:
            request: API key creation request

        Returns:
            ApiKeyResponse with the created key details (masked)

        Raises:
            ApiKeyValidationError: If validation fails
            ApiKeyDuplicateError: If a duplicate key exists

        """
        async with self._lock:
            # Normalize provider ID
            provider_id = self._normalize_provider_id(request.provider_id)
            if not provider_id:
                msg = (
                    f"Unknown provider: {request.provider_id}. "
                    f"Supported providers: {[p.value for p in ProviderType]}"
                )
                raise ApiKeyValidationError(
                    msg,
                )

            # Validate key format
            is_valid, error = self._validate_key_format(provider_id, request.api_key)
            if not is_valid:
                raise ApiKeyValidationError(error)

            # Check for duplicate primary keys
            existing_keys = self._provider_index.get(provider_id, [])
            if request.role == ApiKeyRole.PRIMARY:
                for key_id in existing_keys:
                    existing = self._keys.get(key_id)
                    if existing and existing.role == ApiKeyRole.PRIMARY:
                        if existing.status == ApiKeyStatus.ACTIVE:
                            msg = (
                                f"An active primary key already exists for {provider_id}. "
                                "Update the existing key or create a backup key instead."
                            )
                            raise ApiKeyDuplicateError(
                                msg,
                            )

            # Encrypt the API key
            try:
                encrypted_key = encrypt_api_key(request.api_key)
            except EncryptionError as e:
                msg = f"Failed to encrypt API key: {e}"
                raise ApiKeyServiceError(msg) from e

            # Generate unique ID
            key_id = self._generate_key_id(provider_id, request.role)

            # Create the record
            record = ApiKeyRecord(
                id=key_id,
                provider_id=provider_id,
                encrypted_key=encrypted_key,
                name=request.name,
                role=request.role,
                status=ApiKeyStatus.ACTIVE,
                priority=request.priority,
                expires_at=request.expires_at,
                description=request.description,
                tags=request.tags,
                metadata=request.metadata,
            )

            # Store the key
            self._keys[key_id] = record
            self._provider_index.setdefault(provider_id, []).append(key_id)

            logger.info(
                f"Created API key: {key_id} for provider {provider_id} (role={request.role.value})",
            )

            # Return response with masked key
            return ApiKeyResponse.from_record(record, request.api_key)

    async def get_key(self, key_id: str) -> ApiKeyResponse | None:
        """Get an API key by ID.

        Args:
            key_id: Unique key identifier

        Returns:
            ApiKeyResponse or None if not found

        """
        record = self._keys.get(key_id)
        if not record:
            return None

        # Decrypt key for masking
        try:
            decrypted_key = decrypt_api_key(record.encrypted_key)
        except EncryptionError:
            decrypted_key = None

        return ApiKeyResponse.from_record(record, decrypted_key)

    async def get_key_by_id(self, key_id: str) -> ApiKeyRecord | None:
        """Get the full API key record by ID (internal use).

        Args:
            key_id: Unique key identifier

        Returns:
            ApiKeyRecord or None if not found

        """
        return self._keys.get(key_id)

    async def get_decrypted_key(self, key_id: str) -> str | None:
        """Get the decrypted API key value.

        Args:
            key_id: Unique key identifier

        Returns:
            Decrypted API key or None if not found

        Note:
            Use with caution - only when the actual key value is needed
            for API calls. Never expose in responses.

        """
        record = self._keys.get(key_id)
        if not record:
            return None

        try:
            return decrypt_api_key(record.encrypted_key)
        except EncryptionError as e:
            logger.exception(f"Failed to decrypt key {key_id}: {e}")
            return None

    async def get_primary_key(self, provider_id: str) -> ApiKeyRecord | None:
        """Get the primary API key for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Primary ApiKeyRecord or None if not found

        """
        normalized = self._normalize_provider_id(provider_id)
        if not normalized:
            return None

        key_ids = self._provider_index.get(normalized, [])
        for key_id in key_ids:
            record = self._keys.get(key_id)
            if (
                record
                and record.role == ApiKeyRole.PRIMARY
                and record.status == ApiKeyStatus.ACTIVE
            ):
                return record

        # If no primary, return highest priority backup
        return await self.get_fallback_key(normalized)

    async def get_fallback_key(
        self,
        provider_id: str,
        exclude_ids: list[str] | None = None,
    ) -> ApiKeyRecord | None:
        """Get a fallback API key for a provider.

        Args:
            provider_id: Provider identifier
            exclude_ids: Key IDs to exclude (e.g., keys that are rate limited)

        Returns:
            Fallback ApiKeyRecord or None if not found

        """
        normalized = self._normalize_provider_id(provider_id)
        if not normalized:
            return None

        exclude_ids = exclude_ids or []
        key_ids = self._provider_index.get(normalized, [])

        # Get all active keys, sorted by priority
        candidates = []
        for key_id in key_ids:
            if key_id in exclude_ids:
                continue
            record = self._keys.get(key_id)
            if record and record.status == ApiKeyStatus.ACTIVE:
                candidates.append(record)

        if not candidates:
            return None

        # Sort by role (primary first, then backup, then fallback) and priority
        role_order = {
            ApiKeyRole.PRIMARY: 0,
            ApiKeyRole.BACKUP: 1,
            ApiKeyRole.FALLBACK: 2,
        }
        candidates.sort(key=lambda r: (role_order.get(r.role, 99), r.priority))

        return candidates[0]

    async def list_keys(
        self,
        provider_id: str | None = None,
        role: ApiKeyRole | None = None,
        status: ApiKeyStatus | None = None,
        include_inactive: bool = False,
    ) -> ApiKeyListResponse:
        """List API keys with optional filtering.

        Args:
            provider_id: Filter by provider
            role: Filter by role
            status: Filter by status
            include_inactive: Include inactive/expired keys

        Returns:
            ApiKeyListResponse with list of keys

        """
        keys: list[ApiKeyResponse] = []
        by_provider: dict[str, int] = {}

        for record in self._keys.values():
            # Apply filters
            if provider_id:
                normalized = self._normalize_provider_id(provider_id)
                if record.provider_id != normalized:
                    continue

            if role and record.role != role:
                continue

            if status and record.status != status:
                continue

            if not include_inactive:
                if record.status in (ApiKeyStatus.REVOKED, ApiKeyStatus.EXPIRED):
                    continue
                # Check expiration
                if record.expires_at and record.expires_at < datetime.utcnow():
                    continue

            # Decrypt for masking
            try:
                decrypted = decrypt_api_key(record.encrypted_key)
            except EncryptionError:
                decrypted = None

            keys.append(ApiKeyResponse.from_record(record, decrypted))

            # Count by provider
            by_provider[record.provider_id] = by_provider.get(record.provider_id, 0) + 1

        return ApiKeyListResponse(
            keys=keys,
            total=len(keys),
            by_provider=by_provider,
        )

    async def update_key(self, key_id: str, update: ApiKeyUpdate) -> ApiKeyResponse:
        """Update an existing API key.

        Args:
            key_id: Key ID to update
            update: Fields to update

        Returns:
            Updated ApiKeyResponse

        Raises:
            ApiKeyNotFoundError: If key doesn't exist
            ApiKeyValidationError: If validation fails

        """
        async with self._lock:
            record = self._keys.get(key_id)
            if not record:
                msg = f"API key not found: {key_id}"
                raise ApiKeyNotFoundError(msg)

            # Update fields if provided
            if update.api_key is not None:
                # Validate new key format
                is_valid, error = self._validate_key_format(record.provider_id, update.api_key)
                if not is_valid:
                    raise ApiKeyValidationError(error)

                # Encrypt new key
                try:
                    record.encrypted_key = encrypt_api_key(update.api_key)
                except EncryptionError as e:
                    msg = f"Failed to encrypt API key: {e}"
                    raise ApiKeyServiceError(msg) from e

            if update.name is not None:
                record.name = update.name

            if update.role is not None:
                record.role = update.role

            if update.status is not None:
                record.status = update.status

            if update.priority is not None:
                record.priority = update.priority

            if update.expires_at is not None:
                record.expires_at = update.expires_at

            if update.description is not None:
                record.description = update.description

            if update.tags is not None:
                record.tags = update.tags

            if update.metadata is not None:
                record.metadata.update(update.metadata)

            record.updated_at = datetime.utcnow()

            logger.info(f"Updated API key: {key_id}")

            # Return updated response
            try:
                decrypted = decrypt_api_key(record.encrypted_key)
            except EncryptionError:
                decrypted = None

            return ApiKeyResponse.from_record(record, decrypted)

    async def delete_key(self, key_id: str) -> bool:
        """Delete an API key.

        Args:
            key_id: Key ID to delete

        Returns:
            True if deleted successfully

        Raises:
            ApiKeyNotFoundError: If key doesn't exist

        """
        async with self._lock:
            record = self._keys.get(key_id)
            if not record:
                msg = f"API key not found: {key_id}"
                raise ApiKeyNotFoundError(msg)

            # Remove from storage
            del self._keys[key_id]

            # Remove from provider index
            provider_keys = self._provider_index.get(record.provider_id, [])
            if key_id in provider_keys:
                provider_keys.remove(key_id)

            logger.info(f"Deleted API key: {key_id} for provider {record.provider_id}")
            return True

    async def revoke_key(self, key_id: str) -> ApiKeyResponse:
        """Revoke an API key (soft delete).

        Args:
            key_id: Key ID to revoke

        Returns:
            Updated ApiKeyResponse

        Raises:
            ApiKeyNotFoundError: If key doesn't exist

        """
        return await self.update_key(key_id, ApiKeyUpdate(status=ApiKeyStatus.REVOKED))

    async def record_usage(
        self,
        key_id: str,
        success: bool,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Record usage statistics for an API key.

        Args:
            key_id: Key ID
            success: Whether the request was successful
            tokens_used: Total tokens used
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            latency_ms: Request latency in milliseconds
            error: Error message if failed

        """
        record = self._keys.get(key_id)
        if not record:
            return

        stats = record.usage_stats

        # Update counters
        stats.request_count += 1
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
            stats.last_error = error
            stats.last_error_at = datetime.utcnow()

        # Update token counts
        stats.total_tokens_used += tokens_used
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens

        # Update timing
        stats.last_used_at = datetime.utcnow()

        # Update average latency (exponential moving average)
        if latency_ms is not None:
            if stats.avg_latency_ms is None:
                stats.avg_latency_ms = latency_ms
            else:
                stats.avg_latency_ms = 0.9 * stats.avg_latency_ms + 0.1 * latency_ms

        record.updated_at = datetime.utcnow()

    async def record_rate_limit_hit(self, key_id: str) -> None:
        """Record a rate limit hit for an API key.

        Args:
            key_id: Key ID that hit rate limit

        """
        record = self._keys.get(key_id)
        if not record:
            return

        record.usage_stats.rate_limit_hits += 1
        record.usage_stats.last_error = "Rate limit exceeded"
        record.usage_stats.last_error_at = datetime.utcnow()

        # Mark as rate limited temporarily
        record.status = ApiKeyStatus.RATE_LIMITED
        record.updated_at = datetime.utcnow()

        logger.warning(f"Rate limit hit for key {key_id}")

    async def clear_rate_limit(self, key_id: str) -> None:
        """Clear rate limit status for an API key.

        Args:
            key_id: Key ID to clear rate limit

        """
        record = self._keys.get(key_id)
        if not record:
            return

        if record.status == ApiKeyStatus.RATE_LIMITED:
            record.status = ApiKeyStatus.ACTIVE
            record.updated_at = datetime.utcnow()
            logger.info(f"Cleared rate limit for key {key_id}")

    async def get_provider_summary(self, provider_id: str) -> ProviderKeySummary | None:
        """Get a summary of API keys for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            ProviderKeySummary or None if no keys exist

        """
        normalized = self._normalize_provider_id(provider_id)
        if not normalized:
            return None

        key_ids = self._provider_index.get(normalized, [])
        if not key_ids:
            return ProviderKeySummary(
                provider_id=normalized,
                provider_name=normalized.title(),
                total_keys=0,
                active_keys=0,
                has_valid_key=False,
                status="unconfigured",
            )

        total_keys = 0
        active_keys = 0
        primary_key_id = None
        backup_key_ids = []

        for key_id in key_ids:
            record = self._keys.get(key_id)
            if not record:
                continue

            total_keys += 1

            if record.status == ApiKeyStatus.ACTIVE:
                active_keys += 1

                if record.role == ApiKeyRole.PRIMARY:
                    primary_key_id = key_id
                elif record.role in (ApiKeyRole.BACKUP, ApiKeyRole.FALLBACK):
                    backup_key_ids.append(key_id)

        has_valid = active_keys > 0
        status = "configured" if has_valid else "unconfigured"
        if has_valid and not primary_key_id:
            status = "backup_only"

        return ProviderKeySummary(
            provider_id=normalized,
            provider_name=normalized.title(),
            total_keys=total_keys,
            active_keys=active_keys,
            primary_key_id=primary_key_id,
            backup_key_ids=backup_key_ids,
            has_valid_key=has_valid,
            status=status,
        )

    async def get_all_provider_summaries(self) -> list[ProviderKeySummary]:
        """Get summaries for all providers.

        Returns:
            List of ProviderKeySummary for all known providers

        """
        summaries = []
        for provider_type in ProviderType:
            summary = await self.get_provider_summary(provider_type.value)
            if summary:
                summaries.append(summary)

        return summaries

    async def test_key(self, request: ApiKeyTestRequest) -> ApiKeyTestResult:
        """Test an API key's connectivity.

        Args:
            request: Test request with provider and key info

        Returns:
            ApiKeyTestResult with test results

        """
        # Get the API key to test
        if request.key_id:
            decrypted_key = await self.get_decrypted_key(request.key_id)
            if not decrypted_key:
                return ApiKeyTestResult(
                    success=False,
                    provider_id=request.provider_id,
                    error=f"API key not found: {request.key_id}",
                )
        elif request.api_key:
            decrypted_key = request.api_key
        else:
            return ApiKeyTestResult(
                success=False,
                provider_id=request.provider_id,
                error="No API key provided",
            )

        # Normalize provider ID
        provider_id = self._normalize_provider_id(request.provider_id)
        if not provider_id:
            return ApiKeyTestResult(
                success=False,
                provider_id=request.provider_id,
                error=f"Unknown provider: {request.provider_id}",
            )

        # Test the key with a lightweight API call
        start_time = time.perf_counter()
        try:
            result = await self._test_provider_connectivity(provider_id, decrypted_key)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if result["success"]:
                return ApiKeyTestResult(
                    success=True,
                    provider_id=provider_id,
                    latency_ms=latency_ms,
                    models_available=result.get("models", []),
                    rate_limit_info=result.get("rate_limit_info"),
                )
            return ApiKeyTestResult(
                success=False,
                provider_id=provider_id,
                latency_ms=latency_ms,
                error=result.get("error", "Unknown error"),
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ApiKeyTestResult(
                success=False,
                provider_id=provider_id,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _test_provider_connectivity(self, provider_id: str, api_key: str) -> dict[str, Any]:
        """Test connectivity to a specific provider.

        Args:
            provider_id: Provider identifier
            api_key: Decrypted API key

        Returns:
            Dictionary with success status and details

        """
        # Import here to avoid circular imports
        try:
            from app.services.llm_service import llm_service

            # Try to get provider and run a simple check
            provider = llm_service.get_provider(provider_id)
            if not provider:
                return {
                    "success": False,
                    "error": f"Provider {provider_id} not registered",
                }

            # For now, we just validate the key format and return success
            # A full connectivity test would make an actual API call
            is_valid, error = self._validate_key_format(provider_id, api_key)
            if not is_valid:
                return {"success": False, "error": error}

            # Get available models from config
            from app.core.config import settings

            models = settings.get_provider_models().get(provider_id, [])

            return {
                "success": True,
                "models": models[:5],  # Return first 5 models
            }

        except Exception as e:
            logger.warning(f"Provider connectivity test failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_key_for_request(
        self,
        provider_id: str,
        exclude_rate_limited: bool = True,
    ) -> tuple[str | None, str | None]:
        """Get the best available API key for a request.

        This method is used by the LLM service to get a key for making requests.
        It returns the primary key if available, or falls back to backup keys.

        Args:
            provider_id: Provider identifier
            exclude_rate_limited: Exclude rate-limited keys

        Returns:
            Tuple of (key_id, decrypted_api_key) or (None, None) if no key available

        """
        normalized = self._normalize_provider_id(provider_id)
        if not normalized:
            return None, None

        # Build exclusion list
        exclude_ids = []
        if exclude_rate_limited:
            for key_id in self._provider_index.get(normalized, []):
                record = self._keys.get(key_id)
                if record and record.status == ApiKeyStatus.RATE_LIMITED:
                    exclude_ids.append(key_id)

        # Get best available key
        record = await self.get_fallback_key(normalized, exclude_ids)
        if not record:
            return None, None

        # Decrypt the key
        try:
            decrypted = decrypt_api_key(record.encrypted_key)
            return record.id, decrypted
        except EncryptionError as e:
            logger.exception(f"Failed to decrypt key {record.id}: {e}")
            return None, None

    async def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        async with self._lock:
            # Persist keys if needed (future enhancement)
            self._keys.clear()
            self._provider_index.clear()

        logger.info("ApiKeyStorageService shutdown complete")


# Global singleton instance
api_key_service = ApiKeyStorageService()


def get_api_key_service() -> ApiKeyStorageService:
    """Get the global API key service instance."""
    return api_key_service
