from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# We define a new Base for these models.
# In a fully integrated system, we might import a shared Base.
Base = declarative_base()


class DBLLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    provider = Column(String, nullable=False)  # Store enum value as string
    description = Column(String, nullable=True)
    config = Column(
        JSON, nullable=False, default={}
    )  # Store sanitized config, API keys should be in a secrets manager
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    evasion_tasks = relationship("DBEvasionTask", back_populates="target_model")


class DBEvasionTask(Base):
    __tablename__ = "evasion_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True, nullable=False)  # Celery task ID
    target_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=False)
    initial_prompt = Column(String, nullable=False)
    strategy_chain = Column(JSON, nullable=False)  # List of MetamorphosisStrategyConfig
    success_criteria = Column(String, nullable=False)
    max_attempts = Column(Integer, nullable=False, default=1)
    status = Column(String, default="PENDING")  # Store enum value as string
    final_status = Column(String, nullable=True)  # E.g., "SUCCESS", "FAILURE", "PARTIAL_SUCCESS"
    overall_success = Column(Boolean, default=False)
    failed_reason = Column(String, nullable=True)
    results = Column(JSON, default=[])  # List of EvasionAttemptResult
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)

    # Optional link to a dataset prompt
    jailbreak_prompt_id = Column(Integer, ForeignKey("jailbreak_prompts.id"), nullable=True)
    target_model = relationship("DBLLMModel", back_populates="evasion_tasks")
    jailbreak_prompt = relationship("JailbreakPrompt", back_populates="evasion_tasks")


# New models for integrated datasets
class JailbreakDataset(Base):
    __tablename__ = "jailbreak_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    license = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    prompts = relationship(
        "JailbreakPrompt", back_populates="dataset", cascade="all, delete-orphan"
    )


class JailbreakPrompt(Base):
    __tablename__ = "jailbreak_prompts"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("jailbreak_datasets.id"), nullable=False)
    prompt_text = Column(String, nullable=False)
    jailbreak_type = Column(String, nullable=True)
    is_jailbreak = Column(Boolean, default=True)
    platform = Column(String, nullable=True)
    source = Column(String, nullable=True)
    community = Column(String, nullable=True)
    source_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    dataset = relationship("JailbreakDataset", back_populates="prompts")
    evasion_tasks = relationship("DBEvasionTask", back_populates="jailbreak_prompt")
