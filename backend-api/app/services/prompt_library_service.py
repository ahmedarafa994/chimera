from app.domain.prompt_library_models import PromptTemplate, TemplateMetadata, TemplateVersion
from app.repositories.prompt_library_repository import prompt_library_repository
from app.schemas.prompt_library import (
    CreateTemplateRequest,
    SaveFromCampaignRequest,
    SearchTemplatesRequest,
    TemplateStatsResponse,
    UpdateTemplateRequest,
)


class PromptLibraryService:
    async def create_template(self, user_id: str, request: CreateTemplateRequest) -> PromptTemplate:
        metadata = TemplateMetadata(
            technique_types=request.technique_types,
            vulnerability_types=request.vulnerability_types,
            target_models=request.target_models,
            tags=request.tags,
            custom_data=request.custom_data,
        )

        initial_version = TemplateVersion(
            prompt_text=request.prompt_text,
            created_by=user_id,
            description="Initial version",
        )

        template = PromptTemplate(
            title=request.title,
            description=request.description,
            original_prompt=request.prompt_text,
            current_version_id=initial_version.version_id,
            owner_id=user_id,
            sharing_level=request.sharing_level,
            metadata=metadata,
            versions=[initial_version],
        )

        return await prompt_library_repository.create(template)

    async def get_template(self, template_id: str) -> PromptTemplate | None:
        return await prompt_library_repository.get_by_id(template_id)

    async def update_template(
        self,
        template_id: str,
        request: UpdateTemplateRequest,
    ) -> PromptTemplate | None:
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        if request.title is not None:
            template.title = request.title
        if request.description is not None:
            template.description = request.description
        if request.sharing_level is not None:
            template.sharing_level = request.sharing_level
        if request.status is not None:
            template.status = request.status

        if request.technique_types is not None:
            template.metadata.technique_types = request.technique_types
        if request.vulnerability_types is not None:
            template.metadata.vulnerability_types = request.vulnerability_types
        if request.target_models is not None:
            template.metadata.target_models = request.target_models
        if request.tags is not None:
            template.metadata.tags = request.tags
        if request.custom_data is not None:
            template.metadata.custom_data = request.custom_data

        return await prompt_library_repository.update(template)

    async def delete_template(self, template_id: str) -> bool:
        return await prompt_library_repository.delete(template_id)

    async def search_templates(
        self,
        request: SearchTemplatesRequest,
    ) -> tuple[list[PromptTemplate], int]:
        return await prompt_library_repository.search(
            query=request.query,
            technique_type=request.technique_type,
            vulnerability_type=request.vulnerability_type,
            sharing_level=request.sharing_level,
            tags=request.tags,
            owner_id=request.owner_id,
            min_rating=request.min_rating,
            limit=request.limit,
            offset=request.offset,
        )

    async def get_statistics(self) -> TemplateStatsResponse:
        """Get library-wide statistics."""
        stats = await prompt_library_repository.get_statistics()
        return TemplateStatsResponse(
            total_templates=stats.get("total_templates", 0),
            total_ratings=stats.get("total_ratings", 0),
            avg_effectiveness=stats.get("avg_effectiveness", 0.0),
        )

    async def get_top_rated(self, limit: int = 5) -> list[PromptTemplate]:
        """Get top-rated templates."""
        return await prompt_library_repository.get_top_rated(limit=limit)

    async def save_from_campaign(
        self,
        user_id: str,
        request: SaveFromCampaignRequest,
    ) -> PromptTemplate | None:
        """Save a successful attack from a campaign as a template."""
        # Get the campaign attack data
        campaign_data = await prompt_library_repository.get_campaign_attack(
            campaign_id=request.campaign_id,
            attack_id=request.attack_id,
        )

        if not campaign_data:
            return None

        # Create template from campaign attack
        metadata = TemplateMetadata(
            technique_types=campaign_data.get("technique_types", []),
            vulnerability_types=campaign_data.get("vulnerability_types", []),
            target_models=campaign_data.get("target_models", []),
            tags=["from-campaign", request.campaign_id],
            custom_data={
                "source_campaign_id": request.campaign_id,
                "source_attack_id": request.attack_id,
                "original_success_rate": campaign_data.get("success_rate", 0),
            },
        )

        initial_version = TemplateVersion(
            prompt_text=campaign_data.get("prompt_text", ""),
            created_by=user_id,
            description=f"Imported from campaign {request.campaign_id}",
        )

        template = PromptTemplate(
            title=request.title,
            description=request.description,
            original_prompt=campaign_data.get("prompt_text", ""),
            current_version_id=initial_version.version_id,
            owner_id=user_id,
            sharing_level=request.sharing_level,
            metadata=metadata,
            versions=[initial_version],
        )

        return await prompt_library_repository.create(template)


# Singleton instance
prompt_library_service = PromptLibraryService()
