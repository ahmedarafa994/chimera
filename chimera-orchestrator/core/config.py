"""
Configuration management for Chimera Orchestrator
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int | None = None  # No timeout
    rate_limit: int = 60  # requests per minute
    enabled: bool = True


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator agent."""

    safety_threshold: float = 0.7
    jailbreak_confidence_threshold: float = 0.8
    harmful_keywords: list[str] = field(default_factory=list)
    safe_response_patterns: list[str] = field(default_factory=list)
    use_llm_evaluation: bool = True
    evaluation_model: str = "gpt-4"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator agent."""

    max_concurrent_jobs: int = 10
    job_timeout: int | None = None  # No timeout
    retry_delay: int = 5  # seconds
    max_retries: int = 3
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class QueueConfig:
    """Configuration for the message queue."""

    queue_type: str = "memory"  # memory, priority, redis
    redis_url: str = "redis://localhost:6379"
    max_queue_size: int = 10000
    message_ttl: int = 3600


@dataclass
class APIConfig:
    """Configuration for the REST API."""

    host: str = "0.0.0.0"
    port: int = 8002
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_key: str = ""
    rate_limit: int = 100  # requests per minute


@dataclass
class DatabaseConfig:
    """Configuration for the database."""

    type: str = "sqlite"  # sqlite, postgresql
    url: str = "sqlite:///chimera_orchestrator.db"
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class Config:
    """Main configuration class."""

    # Provider configurations
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)

    # Agent configurations
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    # Infrastructure configurations
    queue: QueueConfig = field(default_factory=QueueConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    datasets_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "imported_data"
    )

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()

        # Load providers
        if "providers" in data:
            for name, provider_data in data["providers"].items():
                config.providers[name] = LLMProviderConfig(name=name, **provider_data)

        # Load evaluator config
        if "evaluator" in data:
            config.evaluator = EvaluatorConfig(**data["evaluator"])

        # Load orchestrator config
        if "orchestrator" in data:
            config.orchestrator = OrchestratorConfig(**data["orchestrator"])

        # Load queue config
        if "queue" in data:
            config.queue = QueueConfig(**data["queue"])

        # Load API config
        if "api" in data:
            config.api = APIConfig(**data["api"])

        # Load database config
        if "database" in data:
            config.database = DatabaseConfig(**data["database"])

        # Load paths
        if "base_path" in data:
            config.base_path = Path(data["base_path"])
        if "datasets_path" in data:
            config.datasets_path = Path(data["datasets_path"])

        # Load logging
        if "log_level" in data:
            config.log_level = data["log_level"]
        if "log_file" in data:
            config.log_file = data["log_file"]

        return config

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()

        # Load providers from environment
        providers = {
            "openai": LLMProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                enabled=bool(os.getenv("OPENAI_API_KEY")),
            ),
            "anthropic": LLMProviderConfig(
                name="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
            ),
            "local": LLMProviderConfig(
                name="local",
                api_key=os.getenv("LOCAL_API_KEY", "admin"),
                base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8005"),
                model=os.getenv("LOCAL_MODEL", "default"),
                enabled=True,
            ),
        }
        config.providers = providers

        # Load API config from environment
        config.api.host = os.getenv("API_HOST", "0.0.0.0")
        config.api.port = int(os.getenv("API_PORT", "8002"))
        config.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        config.api.api_key = os.getenv("API_KEY", "")

        # Load queue config from environment
        config.queue.queue_type = os.getenv("QUEUE_TYPE", "memory")
        config.queue.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Load database config from environment
        config.database.url = os.getenv("DATABASE_URL", "sqlite:///chimera_orchestrator.db")

        # Load logging from environment
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.log_file = os.getenv("LOG_FILE")

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "providers": {
                name: {
                    "api_key": "***" if p.api_key else "",
                    "base_url": p.base_url,
                    "model": p.model,
                    "max_tokens": p.max_tokens,
                    "temperature": p.temperature,
                    "timeout": p.timeout,
                    "rate_limit": p.rate_limit,
                    "enabled": p.enabled,
                }
                for name, p in self.providers.items()
            },
            "evaluator": {
                "safety_threshold": self.evaluator.safety_threshold,
                "jailbreak_confidence_threshold": self.evaluator.jailbreak_confidence_threshold,
                "use_llm_evaluation": self.evaluator.use_llm_evaluation,
                "evaluation_model": self.evaluator.evaluation_model,
            },
            "orchestrator": {
                "max_concurrent_jobs": self.orchestrator.max_concurrent_jobs,
                "job_timeout": self.orchestrator.job_timeout,
                "retry_delay": self.orchestrator.retry_delay,
                "max_retries": self.orchestrator.max_retries,
                "enable_caching": self.orchestrator.enable_caching,
            },
            "queue": {
                "queue_type": self.queue.queue_type,
                "max_queue_size": self.queue.max_queue_size,
            },
            "api": {"host": self.api.host, "port": self.api.port, "debug": self.api.debug},
            "database": {
                "type": self.database.type,
                "url": "***",  # Hide connection string
            },
            "log_level": self.log_level,
        }

    def save_yaml(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration, loading from environment."""
    return Config.from_env()
