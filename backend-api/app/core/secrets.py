"""
Secrets Management Module
Production-grade secrets handling with multiple provider support
"""

import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets providers"""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key"""
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """Environment variable based secrets (development only)"""

    def get_secret(self, key: str) -> str | None:
        value = os.getenv(key)
        # CRIT-004 & HIGH-001 FIX: Enforce required secrets validation at startup
        if value and value.startswith("CHANGE_ME"):
            env = os.getenv("ENVIRONMENT", "development")
            if env == "production":
                raise ValueError(
                    f"CRITICAL: Secret {key} has placeholder value 'CHANGE_ME*' in production. "
                    f"All secrets must be properly configured before starting the application."
                )
            logger.warning(
                f"Secret {key} has not been configured - using placeholder value (development mode only)"
            )
        return value


class AzureKeyVaultProvider(SecretsProvider):
    """Azure Key Vault secrets provider"""

    def __init__(self):
        self.vault_url = os.getenv("AZURE_VAULT_URL")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient

                credential = DefaultAzureCredential()
                self._client = SecretClient(vault_url=self.vault_url, credential=credential)
            except ImportError:
                raise ImportError(
                    "Azure Key Vault SDK not installed. "
                    "Install with: pip install azure-keyvault-secrets azure-identity"
                )
        return self._client

    def get_secret(self, key: str) -> str | None:
        try:
            # Convert environment variable format to Azure Key Vault format
            # e.g., CHIMERA_API_KEY -> chimera-api-key
            vault_key = key.lower().replace("_", "-")
            secret = self.client.get_secret(vault_key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key} from Azure Key Vault: {e}")
            return None


class AWSSecretsManagerProvider(SecretsProvider):
    """AWS Secrets Manager provider"""

    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.secret_name = os.getenv("AWS_SECRET_NAME", "chimera/secrets")
        self._client = None
        self._secrets_cache = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("secretsmanager", region_name=self.region)
            except ImportError:
                raise ImportError("AWS SDK not installed. Install with: pip install boto3")
        return self._client

    def _load_secrets(self):
        if self._secrets_cache is None:
            import json

            try:
                response = self.client.get_secret_value(SecretId=self.secret_name)
                self._secrets_cache = json.loads(response["SecretString"])
            except Exception as e:
                logger.error(f"Failed to load secrets from AWS Secrets Manager: {e}")
                self._secrets_cache = {}
        return self._secrets_cache

    def get_secret(self, key: str) -> str | None:
        secrets = self._load_secrets()
        return secrets.get(key)


class HashiCorpVaultProvider(SecretsProvider):
    """HashiCorp Vault secrets provider"""

    def __init__(self):
        self.vault_addr = os.getenv("VAULT_ADDR")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.mount_point = os.getenv("VAULT_MOUNT_POINT", "secret")
        self.secret_path = os.getenv("VAULT_SECRET_PATH", "chimera")
        self._client = None
        self._secrets_cache = None

    @property
    def client(self):
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(url=self.vault_addr, token=self.vault_token)
            except ImportError:
                raise ImportError(
                    "HashiCorp Vault SDK not installed. Install with: pip install hvac"
                )
        return self._client

    def _load_secrets(self):
        if self._secrets_cache is None:
            try:
                response = self.client.secrets.kv.v2.read_secret_version(
                    mount_point=self.mount_point, path=self.secret_path
                )
                self._secrets_cache = response["data"]["data"]
            except Exception as e:
                logger.error(f"Failed to load secrets from HashiCorp Vault: {e}")
                self._secrets_cache = {}
        return self._secrets_cache

    def get_secret(self, key: str) -> str | None:
        secrets = self._load_secrets()
        return secrets.get(key)


class SecretsManager:
    """
    Centralized secrets management with multiple provider support.

    Usage:
        secrets = SecretsManager()
        api_key = secrets.get_secret("CHIMERA_API_KEY")
    """

    _instance: ClassVar[Optional["SecretsManager"]] = None
    _providers: ClassVar[dict[str, type[SecretsProvider]]] = {
        "env": EnvironmentSecretsProvider,
        "azure": AzureKeyVaultProvider,
        "aws": AWSSecretsManagerProvider,
        "vault": HashiCorpVaultProvider,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        provider_name = os.getenv("SECRETS_PROVIDER", "env").lower()

        if provider_name not in self._providers:
            logger.warning(f"Unknown secrets provider: {provider_name}, falling back to env")
            provider_name = "env"

        self.provider = self._providers[provider_name]()
        self.provider_name = provider_name
        self._initialized = True

        logger.info(f"SecretsManager initialized with provider: {provider_name}")

    @lru_cache(maxsize=100)
    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """
        Get a secret value by key.

        Args:
            key: The secret key (e.g., "CHIMERA_API_KEY")
            default: Default value if secret not found

        Returns:
            The secret value or default
        """
        value = self.provider.get_secret(key)

        if value is None:
            if default is not None:
                return default
            logger.warning(f"Secret {key} not found and no default provided")

        return value

    def get_required_secret(self, key: str) -> str:
        """
        Get a required secret value. Raises if not found.

        Args:
            key: The secret key

        Returns:
            The secret value

        Raises:
            ValueError: If the secret is not found
        """
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Required secret {key} not found")
        return value

    def clear_cache(self):
        """Clear the secrets cache"""
        self.get_secret.cache_clear()


# Convenience function for getting secrets
@lru_cache
def get_secrets_manager() -> SecretsManager:
    """Get the singleton SecretsManager instance"""
    return SecretsManager()


def get_secret(key: str, default: str | None = None) -> str | None:
    """Convenience function to get a secret"""
    return get_secrets_manager().get_secret(key, default)
