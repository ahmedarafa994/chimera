"""Endpoints for dataset import and access"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.core.database import SyncSessionFactory
from app.crud import crud_jailbreak
from app.schemas.jailbreak import JailbreakDatasetRead
from app.services.jailbreak_importer import JailbreakImporter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/datasets", response_model=list[JailbreakDatasetRead])
def list_datasets(limit: int = 100, offset: int = 0):
    """List datasets"""
    with SyncSessionFactory() as db:
        ds = crud_jailbreak.get_datasets(db, limit=limit, offset=offset)
        return ds


@router.get("/datasets/{dataset_id}/prompts")
def get_prompts(dataset_id: int, limit: int = 100, offset: int = 0):
    """Get prompts for a dataset"""
    with SyncSessionFactory() as db:
        prompts = crud_jailbreak.get_prompts(db, dataset_id=dataset_id, limit=limit, offset=offset)
        return prompts


@router.post("/datasets/import")
def import_dataset(
    filename: str = Query(..., description="Relative filename under imported_data"),
    dataset_name: str | None = Query(None),
    max_lines: int | None = Query(None),
):
    """Import a dataset file (csv or jsonl) from the repository `imported_data/` path."""
    base_path = Path(__file__).resolve().parents[3] / "imported_data"
    file_path = base_path / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    name = dataset_name or file_path.stem

    with SyncSessionFactory() as db:
        importer = JailbreakImporter(db)
        if file_path.suffix.lower() == ".csv":
            count = importer.import_csv(file_path, name)
        else:
            count = importer.import_jsonl(file_path, name, max_lines=max_lines)

    return {"imported": count, "dataset": name}
