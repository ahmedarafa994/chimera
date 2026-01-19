import json
import os
from datetime import datetime
from typing import Any

from app.domain.prompt_library_models import (
    PromptTemplate,
    SharingLevel,
    TechniqueType,
    VulnerabilityType,
)


class RepositoryError(Exception):
    """Base exception for repository errors"""

    pass


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found"""

    pass


class DuplicateEntityError(RepositoryError):
    """Raised when an entity already exists"""

    pass


class ValidationError(RepositoryError):
    """Raised when data validation fails"""

    pass


class PromptLibraryRepository:
    def __init__(self, storage_path: str = "data/prompt_library.json"):
        self.storage_path = storage_path
        self._templates: dict[str, PromptTemplate] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                for item in data:
                    template = PromptTemplate(**item)
                    self._templates[template.id] = template
        except Exception as e:
            print(f"Error loading prompt library: {e}")

    def _save_to_disk(self):
        try:
            data = [t.dict() for t in self._templates.values()]
            with open(self.storage_path, "w") as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            print(f"Error saving prompt library: {e}")

    async def get_by_id(self, template_id: str) -> PromptTemplate | None:
        return self._templates.get(template_id)

    async def create(self, template: PromptTemplate) -> PromptTemplate:
        if template.id in self._templates:
            raise DuplicateEntityError(f"Template with id {template.id} already exists")
        self._templates[template.id] = template
        self._save_to_disk()
        return template

    async def update(self, template: PromptTemplate) -> PromptTemplate:
        if template.id not in self._templates:
            raise EntityNotFoundError(f"Template with id {template.id} not found")
        template.updated_at = datetime.utcnow()
        self._templates[template.id] = template
        self._save_to_disk()
        return template

    async def delete(self, template_id: str) -> bool:
        if template_id in self._templates:
            del self._templates[template_id]
            self._save_to_disk()
            return True
        return False

    async def search(
        self,
        query: str | None = None,
        technique_type: TechniqueType | None = None,
        vulnerability_type: VulnerabilityType | None = None,
        sharing_level: SharingLevel | None = None,
        tags: list[str] | None = None,
        owner_id: str | None = None,
        min_rating: float | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[PromptTemplate], int]:
        results = list(self._templates.values())

        if query:
            q = query.lower()
            results = [
                t
                for t in results
                if q in t.title.lower()
                or q in t.description.lower()
                or q in t.original_prompt.lower()
            ]

        if technique_type:
            results = [t for t in results if technique_type in t.metadata.technique_types]

        if vulnerability_type:
            results = [t for t in results if vulnerability_type in t.metadata.vulnerability_types]

        if sharing_level:
            results = [t for t in results if t.sharing_level == sharing_level]

        if tags:
            tag_set = set(tags)
            results = [t for t in results if tag_set.intersection(set(t.metadata.tags))]

        if owner_id:
            results = [t for t in results if t.owner_id == owner_id]

        if min_rating:
            results = [t for t in results if t.avg_rating >= min_rating]

        # Sort by updated_at desc
        results.sort(key=lambda t: t.updated_at, reverse=True)

        total = len(results)
        paged_results = results[offset : offset + limit]

        return paged_results, total

    async def get_statistics(self) -> dict[str, Any]:
        """Get library-wide statistics"""
        templates = list(self._templates.values())
        total_templates = len(templates)

        total_ratings = sum(t.total_ratings for t in templates)

        # Calculate average effectiveness across all templates
        if templates:
            avg_effectiveness = sum(t.effectiveness_score for t in templates) / len(templates)
        else:
            avg_effectiveness = 0.0

        return {
            "total_templates": total_templates,
            "total_ratings": total_ratings,
            "avg_effectiveness": avg_effectiveness,
        }

    async def get_top_rated(self, limit: int = 5) -> list[PromptTemplate]:
        """Get top-rated templates sorted by avg_rating"""
        templates = list(self._templates.values())
        # Filter to only public/team templates and sort by rating
        public_templates = [
            t for t in templates if t.sharing_level in [SharingLevel.PUBLIC, SharingLevel.TEAM]
        ]
        public_templates.sort(key=lambda t: (t.avg_rating, t.total_ratings), reverse=True)
        return public_templates[:limit]

    async def get_campaign_attack(
        self, campaign_id: str, attack_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get campaign attack data for importing as a template"""
        # This would typically query the campaigns table
        # For now, return mock data to allow the endpoint to function
        # In production, integrate with the actual campaign service
        return {
            "prompt_text": f"Attack prompt from campaign {campaign_id}",
            "technique_types": [],
            "vulnerability_types": [],
            "target_models": [],
            "success_rate": 0.0,
        }


# Singleton instance
prompt_library_repository = PromptLibraryRepository()


def get_prompt_library_repository() -> PromptLibraryRepository:
    return prompt_library_repository
