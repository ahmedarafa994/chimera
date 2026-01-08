import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import SyncSessionFactory
from app.crud import llm_crud
from app.schemas.api_schemas import LLMModel, LLMModelCreate

logger = logging.getLogger(__name__)


def get_sync_db():
    db = SyncSessionFactory()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


@router.post("/target-models/", response_model=LLMModel, status_code=status.HTTP_201_CREATED)
def create_llm_model_endpoint(llm_model: LLMModelCreate, db: Session = Depends(get_sync_db)):
    """
    Register a new target LLM model with the platform.
    """
    db_model = llm_crud.get_llm_model_by_name(db, name=llm_model.name)
    if db_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"LLM Model with name '{llm_model.name}' already exists.",
        )

    logger.info(f"Registering new LLM model: {llm_model.name} ({llm_model.provider})")
    created_model = llm_crud.create_llm_model(db=db, llm_model=llm_model)

    return LLMModel.model_validate(created_model)


@router.get("/target-models/", response_model=list[LLMModel])
def read_llm_models_endpoint(skip: int = 0, limit: int = 100, db: Session = Depends(get_sync_db)):
    """
    Retrieve a list of all registered LLM models.
    """
    models = llm_crud.get_llm_models(db, skip=skip, limit=limit)
    return [LLMModel.model_validate(model) for model in models]


@router.get("/target-models/{model_id}", response_model=LLMModel)
def read_llm_model_endpoint(model_id: int, db: Session = Depends(get_sync_db)):
    """
    Retrieve details for a specific LLM model by ID.
    """
    db_model = llm_crud.get_llm_model(db, model_id=model_id)
    if db_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM Model not found")
    return LLMModel.model_validate(db_model)


@router.delete("/target-models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_llm_model_endpoint(model_id: int, db: Session = Depends(get_sync_db)):
    """
    Delete an LLM model by ID.
    """
    db_model = llm_crud.delete_llm_model(db, model_id=model_id)
    if db_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM Model not found")
    logger.info(f"Deleted LLM model with ID: {model_id}")
    return
