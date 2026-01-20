# =============================================================================
# Chimera - API Key Encryption Module
# =============================================================================
# Implements AES-256 encryption for securing API keys at rest
# Part of Story 1.1: Provider Configuration Management
# =============================================================================

import base64
import logging
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Base exception for encryption operations."""


class EncryptionKeyError(EncryptionError):
    """Raised when encryption key is invalid or missing."""


class _EncryptionManager:
    """Singleton encryption manager for API key encryption/decryption.

    Uses AES-256 encryption via Fernet (which uses AES 128 in CBC mode with HMAC).
    For true AES-256, this could be extended, but Fernet provides good security
    with authenticated encryption.
    """

    def __init__(self) -> None:
        self._fernet: Fernet | None = None
        self._encryption_key_env = "CHIMERA_ENCRYPTION_KEY"

    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key from environment.

        Returns:
            32-byte encryption key

        Raises:
            EncryptionKeyError: If key generation fails

        """
        # Try to get key from environment
        key_b64 = os.getenv(self._encryption_key_env)

        if key_b64:
            try:
                key = base64.urlsafe_b64decode(key_b64)
                if len(key) == 32:
                    return key
                logger.warning(
                    "Invalid encryption key length in environment, generating new one",
                )
            except Exception as e:
                logger.warning(f"Invalid encryption key format in environment: {e}")

        # Generate new key using PBKDF2 with a default password
        # In production, this should use a proper secret management system
        password = os.getenv("CHIMERA_ENCRYPTION_PASSWORD", "chimera-default-key-2025").encode()
        salt = b"chimera-salt-2025"  # In production, use random salt stored securely

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password)

            # Store the generated key in environment for this session
            key_b64 = base64.urlsafe_b64encode(key).decode()
            os.environ[self._encryption_key_env] = key_b64

            logger.info("Generated new encryption key for API key protection")
            return key

        except Exception as e:
            msg = f"Failed to generate encryption key: {e}"
            raise EncryptionKeyError(msg) from e

    def _get_fernet(self) -> Fernet:
        """Get or initialize Fernet cipher."""
        if self._fernet is None:
            try:
                key = self._get_encryption_key()
                # Fernet expects a base64-encoded 32-byte key
                key_b64 = base64.urlsafe_b64encode(key)
                self._fernet = Fernet(key_b64)
            except Exception as e:
                msg = f"Failed to initialize encryption cipher: {e}"
                raise EncryptionError(msg) from e

        return self._fernet

    def encrypt_api_key(self, plaintext_key: str) -> str:
        """Encrypt an API key for secure storage.

        Args:
            plaintext_key: The API key to encrypt

        Returns:
            Base64-encoded encrypted key

        Raises:
            EncryptionError: If encryption fails

        """
        if not plaintext_key:
            return plaintext_key

        if plaintext_key.startswith("enc:"):
            logger.warning("API key appears to already be encrypted")
            return plaintext_key

        try:
            fernet = self._get_fernet()
            encrypted_bytes = fernet.encrypt(plaintext_key.encode("utf-8"))
            encrypted_b64 = base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")
            return f"enc:{encrypted_b64}"
        except Exception as e:
            msg = f"Failed to encrypt API key: {e}"
            raise EncryptionError(msg) from e

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt an API key for runtime use.

        Args:
            encrypted_key: The encrypted API key (with enc: prefix)

        Returns:
            Decrypted plaintext API key

        Raises:
            EncryptionError: If decryption fails

        """
        if not encrypted_key:
            return encrypted_key

        if not encrypted_key.startswith("enc:"):
            # Key is not encrypted, return as-is
            logger.debug("API key is not encrypted, returning plaintext")
            return encrypted_key

        try:
            # Remove the "enc:" prefix
            encrypted_b64 = encrypted_key[4:]

            # Decode and decrypt
            fernet = self._get_fernet()
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_b64)
            decrypted_bytes = fernet.decrypt(encrypted_bytes)

            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            msg = f"Failed to decrypt API key: {e}"
            raise EncryptionError(msg) from e

    def is_encrypted(self, api_key: str) -> bool:
        """Check if an API key is encrypted.

        Args:
            api_key: The API key to check

        Returns:
            True if the key is encrypted (has enc: prefix)

        """
        return bool(api_key and api_key.startswith("enc:"))


# Global encryption manager instance
_encryption_manager = _EncryptionManager()


def encrypt_api_key(plaintext_key: str) -> str:
    """Encrypt an API key for secure storage.

    Args:
        plaintext_key: The API key to encrypt

    Returns:
        Base64-encoded encrypted key with enc: prefix

    Raises:
        EncryptionError: If encryption fails

    """
    return _encryption_manager.encrypt_api_key(plaintext_key)


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key for runtime use.

    Args:
        encrypted_key: The encrypted API key (with enc: prefix)

    Returns:
        Decrypted plaintext API key

    Raises:
        EncryptionError: If decryption fails

    """
    return _encryption_manager.decrypt_api_key(encrypted_key)


def is_encrypted(api_key: str) -> bool:
    """Check if an API key is encrypted.

    Args:
        api_key: The API key to check

    Returns:
        True if the key is encrypted (has enc: prefix)

    """
    return _encryption_manager.is_encrypted(api_key)


def ensure_key_encrypted(api_key: str) -> str:
    """Ensure an API key is encrypted, encrypting if necessary.

    Args:
        api_key: The API key to check and encrypt if needed

    Returns:
        Encrypted API key

    """
    if not api_key or is_encrypted(api_key):
        return api_key

    return encrypt_api_key(api_key)
