from app.domain.prompt_library_models import PromptTemplate, TemplateVersion
from app.repositories.prompt_library_repository import prompt_library_repository
from app.schemas.prompt_library import CreateVersionRequest


class TemplateVersionService:
    async def create_version(
        self,
        template_id: str,
        user_id: str,
        request: CreateVersionRequest,
    ) -> PromptTemplate | None:
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        new_version = TemplateVersion(
            parent_version_id=template.current_version_id,
            prompt_text=request.prompt_text,
            description=request.description,
            created_by=user_id,
            metadata_overrides=request.metadata_overrides,
        )

        template.versions.append(new_version)
        template.current_version_id = new_version.version_id

        return await prompt_library_repository.update(template)

    async def get_version(self, template_id: str, version_id: str) -> TemplateVersion | None:
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        for version in template.versions:
            if version.version_id == version_id:
                return version
        return None

    async def restore_version(
        self,
        template_id: str,
        version_id: str,
        user_id: str,
    ) -> PromptTemplate | None:
        """Restore a previous version of a template."""
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        # Find the version to restore
        version_to_restore = None
        for version in template.versions:
            if version.version_id == version_id:
                version_to_restore = version
                break

        if not version_to_restore:
            return None

        # Create a new version based on the old one
        new_version = TemplateVersion(
            parent_version_id=template.current_version_id,
            prompt_text=version_to_restore.prompt_text,
            description=f"Restored from version {version_id}",
            created_by=user_id,
            metadata_overrides=version_to_restore.metadata_overrides,
        )

        template.versions.append(new_version)
        template.current_version_id = new_version.version_id

        return await prompt_library_repository.update(template)


# Singleton instance
template_version_service = TemplateVersionService()
