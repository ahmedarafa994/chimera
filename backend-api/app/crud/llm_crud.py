from sqlalchemy.orm import Session

from app.db.models import DBLLMModel
from app.schemas.api_schemas import LLMModelCreate


def create_llm_model(db: Session, llm_model: LLMModelCreate) -> DBLLMModel:
    # IMPORTANT: In a real application, API keys should NOT be stored directly in the database
    # or even passed around in clear text. Use a secrets management system.
    # For this academic research PoC, we'll store a sanitized config in the DB.
    db_llm_model = DBLLMModel(
        name=llm_model.name,
        provider=llm_model.provider.value,
        description=llm_model.description,
        # Store a copy of the config without the API key for logging/metadata if needed
        config={k: v for k, v in llm_model.config.items() if k.lower() not in ["api_key", "token"]},
    )
    db.add(db_llm_model)
    db.commit()
    db.refresh(db_llm_model)
    return db_llm_model


def get_llm_model(db: Session, model_id: int) -> DBLLMModel | None:
    return db.query(DBLLMModel).filter(DBLLMModel.id == model_id).first()


def get_llm_model_by_name(db: Session, name: str) -> DBLLMModel | None:
    return db.query(DBLLMModel).filter(DBLLMModel.name == name).first()


def get_llm_models(db: Session, skip: int = 0, limit: int = 100) -> list[DBLLMModel]:
    return db.query(DBLLMModel).offset(skip).limit(limit).all()


def delete_llm_model(db: Session, model_id: int):
    db_model = db.query(DBLLMModel).filter(DBLLMModel.id == model_id).first()
    if db_model:
        db.delete(db_model)
        db.commit()
    return db_model
