"""
Jailbreak Dataset and Prompt Schemas

Pydantic models for jailbreak dataset and prompt entities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JailbreakDatasetBase(BaseModel):
    """Base schema for jailbreak datasets."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    source_url: Optional[str] = Field(None, description="Source URL")
    license: Optional[str] = Field(None, description="Dataset license")


class JailbreakDatasetCreate(JailbreakDatasetBase):
    """Schema for creating a new jailbreak dataset."""
    pass


class JailbreakDatasetRead(JailbreakDatasetBase):
    """Schema for reading a jailbreak dataset."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JailbreakPromptBase(BaseModel):
    """Base schema for jailbreak prompts."""
    prompt_text: str = Field(..., description="The actual prompt text")
    jailbreak_type: Optional[str] = Field(None, description="Type of jailbreak")
    is_jailbreak: bool = Field(True, description="Whether this is a jailbreak prompt")
    platform: Optional[str] = Field(None, description="Target platform")
    source: Optional[str] = Field(None, description="Source of the prompt")
    community: Optional[str] = Field(None, description="Community source")
    source_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class JailbreakPromptCreate(JailbreakPromptBase):
    """Schema for creating a new jailbreak prompt."""
    dataset_id: int = Field(..., description="ID of the parent dataset")


class JailbreakPromptRead(JailbreakPromptBase):
    """Schema for reading a jailbreak prompt."""
    id: int
    dataset_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JailbreakPromptBulkCreate(BaseModel):
    """Schema for bulk creating jailbreak prompts."""
    dataset_id: int = Field(..., description="ID of the parent dataset")
    prompts: List[JailbreakPromptBase] = Field(..., description="List of prompts to create")


class JailbreakDatasetWithPrompts(JailbreakDatasetRead):
    """Schema for a jailbreak dataset with its prompts."""
    prompts: List[JailbreakPromptRead] = Field(default_factory=list)