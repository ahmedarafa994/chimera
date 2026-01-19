import pytest

from app.domain.prompt_library_models import SharingLevel, TechniqueType, VulnerabilityType
from app.schemas.prompt_library import CreateTemplateRequest, SearchTemplatesRequest
from app.services.prompt_library_service import prompt_library_service


@pytest.mark.asyncio
async def test_create_template():
    request = CreateTemplateRequest(
        title="Test Template",
        description="A test description for the template.",
        prompt_text="Hello world prompt",
        technique_types=[TechniqueType.MANUAL],
        vulnerability_types=[VulnerabilityType.OTHER],
        sharing_level=SharingLevel.PRIVATE,
        tags=["test"],
    )

    template = await prompt_library_service.create_template("test-user", request)

    assert template.title == "Test Template"
    assert template.owner_id == "test-user"
    assert template.metadata.technique_types == [TechniqueType.MANUAL]
    assert len(template.versions) == 1
    assert template.versions[0].prompt_text == "Hello world prompt"


@pytest.mark.asyncio
async def test_search_templates():
    # Setup - ensure we have something to search
    request = CreateTemplateRequest(
        title="Searchable Template",
        description="Unique description",
        prompt_text="Unique prompt",
        technique_types=[TechniqueType.GPTFUZZ],
        vulnerability_types=[VulnerabilityType.JAILBREAK],
        sharing_level=SharingLevel.PUBLIC,
    )
    await prompt_library_service.create_template("admin", request)

    # Search by query
    search_req = SearchTemplatesRequest(query="Searchable", limit=10, offset=0)
    items, total = await prompt_library_service.search_templates(search_req)

    assert total >= 1
    assert any(t.title == "Searchable Template" for t in items)

    # Search by technique
    search_req = SearchTemplatesRequest(technique_type=TechniqueType.GPTFUZZ, limit=10, offset=0)
    items, total = await prompt_library_service.search_templates(search_req)
    assert any(t.title == "Searchable Template" for t in items)
