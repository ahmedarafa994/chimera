"""
Jailbreak Dataset and Prompt CRUD Operations

Provides database operations for jailbreak datasets and prompts including:
- Creating datasets and prompts
- Retrieving datasets and prompts
- Bulk operations for prompt insertion
- Listing and filtering operations
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import JailbreakDataset, JailbreakPrompt


def create_dataset(
    db: Session,
    name: str,
    description: Optional[str] = None,
    source_url: Optional[str] = None,
    license: Optional[str] = None,
) -> JailbreakDataset:
    """Create a new jailbreak dataset."""
    db_dataset = JailbreakDataset(
        name=name,
        description=description,
        source_url=source_url,
        license=license,
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_dataset_by_name(db: Session, name: str) -> Optional[JailbreakDataset]:
    """Get a jailbreak dataset by name."""
    return db.query(JailbreakDataset).filter(JailbreakDataset.name == name).first()


def get_datasets(
    db: Session, skip: int = 0, limit: int = 100
) -> List[JailbreakDataset]:
    """Get all jailbreak datasets with pagination."""
    return (
        db.query(JailbreakDataset)
        .order_by(desc(JailbreakDataset.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_prompt(
    db: Session,
    dataset_id: int,
    prompt_text: str,
    jailbreak_type: Optional[str] = None,
    is_jailbreak: bool = True,
    platform: Optional[str] = None,
    source: Optional[str] = None,
    community: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> JailbreakPrompt:
    """Create a new jailbreak prompt."""
    db_prompt = JailbreakPrompt(
        dataset_id=dataset_id,
        prompt_text=prompt_text,
        jailbreak_type=jailbreak_type,
        is_jailbreak=is_jailbreak,
        platform=platform,
        source=source,
        community=community,
        source_metadata=source_metadata,
    )
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt


def get_prompts(
    db: Session,
    dataset_id: Optional[int] = None,
    jailbreak_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[JailbreakPrompt]:
    """Get jailbreak prompts with optional filtering."""
    query = db.query(JailbreakPrompt)

    if dataset_id is not None:
        query = query.filter(JailbreakPrompt.dataset_id == dataset_id)

    if jailbreak_type is not None:
        query = query.filter(JailbreakPrompt.jailbreak_type == jailbreak_type)

    return (
        query.order_by(desc(JailbreakPrompt.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def bulk_insert_prompts(
    db: Session, prompts_data: List[Dict[str, Any]]
) -> List[JailbreakPrompt]:
    """Bulk insert jailbreak prompts for efficiency."""
    db_prompts = [JailbreakPrompt(**prompt_data) for prompt_data in prompts_data]
    db.add_all(db_prompts)
    db.commit()

    # Refresh all objects to get their IDs and updated fields
    for prompt in db_prompts:
        db.refresh(prompt)

    return db_prompts