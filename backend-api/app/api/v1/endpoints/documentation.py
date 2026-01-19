"""
Interactive Help & Documentation Hub Endpoints

Phase 2 feature for competitive differentiation:
- In-app searchable documentation
- Contextual help and best practices
- functional guidelines and legal considerations
- Video tutorials and getting started guides
"""

from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.core.observability import get_logger
from app.db.models import User

logger = get_logger("chimera.api.documentation")
router = APIRouter()


# Documentation Models
class DocumentationType(str, Enum):
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    FAQ = "faq"
    functional_GUIDELINES = "functional_guidelines"
    LEGAL = "legal"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"


class DocumentationCategory(str, Enum):
    GETTING_STARTED = "getting_started"
    ATTACK_TECHNIQUES = "attack_techniques"
    PROVIDERS = "providers"
    REPORTING = "reporting"
    COLLABORATION = "collaboration"
    ADVANCED = "advanced"
    ETHICS_LEGAL = "ethics_legal"
    API = "api"


class DocumentationItem(BaseModel):
    """Documentation content item"""

    id: str
    title: str
    content: str
    type: DocumentationType
    category: DocumentationCategory

    # Metadata
    tags: list[str] = Field(default_factory=list)
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    estimated_read_time: int = 5  # minutes

    # Navigation
    parent_id: str | None = None
    order: int = 0

    # Content structure
    sections: list[dict[str, Any]] = Field(default_factory=list)
    related_articles: list[str] = Field(default_factory=list)

    # Media
    video_url: str | None = None
    images: list[str] = Field(default_factory=list)

    # Tracking
    views: int = 0
    last_updated: str
    author: str = "Chimera Team"


class VideoTutorial(BaseModel):
    """Video tutorial information"""

    id: str
    title: str
    description: str
    video_url: str
    thumbnail_url: str | None = None
    duration: int  # seconds
    category: DocumentationCategory
    tags: list[str] = Field(default_factory=list)
    transcript: str | None = None
    created_at: str
    views: int = 0


class SearchRequest(BaseModel):
    """Search documentation request"""

    query: str = Field(..., min_length=1, max_length=200)
    category: DocumentationCategory | None = None
    type: DocumentationType | None = None
    difficulty_level: str | None = None
    tags: list[str] | None = None


class SearchResult(BaseModel):
    """Documentation search result"""

    item: DocumentationItem
    relevance_score: float
    matching_sections: list[str] = Field(default_factory=list)
    highlight_text: str


class DocumentationListResponse(BaseModel):
    """Response for documentation listing"""

    items: list[DocumentationItem]
    categories: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class HelpContext(BaseModel):
    """Contextual help request"""

    page_url: str
    user_action: str | None = None
    feature: str | None = None
    error_code: str | None = None


# In-memory documentation storage (in production, would use database)
documentation_items: dict[str, DocumentationItem] = {}
video_tutorials: dict[str, VideoTutorial] = {}


# Initialize sample documentation data
def initialize_documentation():
    """Initialize sample documentation content"""

    # Getting Started Guide
    getting_started = DocumentationItem(
        id="getting-started-overview",
        title="Getting Started with Chimera",
        content="""
# Welcome to Chimera

Chimera is a comprehensive AI-powered prompt optimization, jailbreak research, and red-teaming platform. This guide will help you get started with your first security assessment.

## What You'll Learn

1. Setting up your API keys
2. Running your first assessment
3. Understanding results
4. Generating reports

## Prerequisites

- Basic understanding of AI/LLM security concepts
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, or DeepSeek)
- Appropriate authorization for security testing

## Next Steps

Follow the onboarding wizard to set up your account and API keys, then proceed to run your first assessment using our quick start templates.
        """,
        type=DocumentationType.TUTORIAL,
        category=DocumentationCategory.GETTING_STARTED,
        tags=["onboarding", "setup", "beginner"],
        difficulty_level="beginner",
        estimated_read_time=5,
        sections=[
            {"title": "Introduction", "anchor": "introduction"},
            {"title": "Prerequisites", "anchor": "prerequisites"},
            {"title": "Next Steps", "anchor": "next-steps"},
        ],
        last_updated="2024-01-16T12:00:00Z",
    )

    # functional Guidelines
    functional_guidelines = DocumentationItem(
        id="functional-guidelines",
        title="functional Guidelines for AI Security Testing",
        content="""
# functional Guidelines

## Core Principles

1. **Authorized Testing Only**: Only test systems you own or have explicit permission to test
2. **Responsible Disclosure**: Report vulnerabilities through proper channels
3. **No Harm**: Never use findings to cause harm or damage
4. **Educational Purpose**: Focus on defense improvement and education

## Best Practices

### Before Testing
- Obtain written authorization
- Define scope and boundaries
- Establish communication channels
- Plan for responsible disclosure

### During Testing
- Stay within authorized scope
- Document findings carefully
- Avoid unnecessary disruption
- Protect sensitive data

### After Testing
- Report findings promptly
- Provide clear remediation guidance
- Support remediation efforts
- Follow up on fixes

## Legal Considerations

Always consult with legal counsel before conducting security assessments. Laws vary by jurisdiction and unauthorized testing may be complex.
        """,
        type=DocumentationType.functional_GUIDELINES,
        category=DocumentationCategory.ETHICS_LEGAL,
        tags=["ethics", "legal", "guidelines", "responsible", "disclosure"],
        difficulty_level="beginner",
        estimated_read_time=8,
        last_updated="2024-01-16T12:00:00Z",
    )

    # Attack Techniques Guide
    attack_techniques = DocumentationItem(
        id="attack-techniques-overview",
        title="Understanding Attack Techniques",
        content="""
# Attack Techniques Overview

Chimera includes multiple advanced attack methodologies for comprehensive LLM security testing.

## Available Techniques

### AutoDAN
Advanced reasoning-based attacks that leverage model understanding to generate effective prompts.

### GPTFuzz
Evolutionary fuzzing approach that mutates prompts based on success feedback.

### HouYi
Prompt optimization technique focused on finding minimal effective variations.

### Gradient-Based Methods
Direct optimization techniques including HotFlip and GCG for token-level attacks.

## Choosing the Right Technique

- **AutoDAN**: Best for complex reasoning tasks and sophisticated models
- **GPTFuzz**: Effective for broad coverage and mutation-based discovery
- **HouYi**: Ideal for prompt optimization and minimal changes
- **Gradient Methods**: Most effective for white-box scenarios

## Combination Strategies

For comprehensive testing, use multiple techniques in sequence to maximize coverage and effectiveness.
        """,
        type=DocumentationType.GUIDE,
        category=DocumentationCategory.ATTACK_TECHNIQUES,
        tags=["autodan", "gptfuzz", "houyi", "gradient", "techniques"],
        difficulty_level="intermediate",
        estimated_read_time=10,
        last_updated="2024-01-16T12:00:00Z",
    )

    documentation_items[getting_started.id] = getting_started
    documentation_items[functional_guidelines.id] = functional_guidelines
    documentation_items[attack_techniques.id] = attack_techniques


# Initialize documentation on module load
initialize_documentation()


@router.get("/", response_model=DocumentationListResponse)
async def list_documentation(
    category: DocumentationCategory | None = Query(None, description="Filter by category"),
    type: DocumentationType | None = Query(None, description="Filter by type"),
    difficulty: str | None = Query(None, description="Filter by difficulty level"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
):
    """List documentation items with filtering and pagination"""
    try:
        # Get all documentation items
        items = list(documentation_items.values())

        # Apply filters
        if category:
            items = [item for item in items if item.category == category]

        if type:
            items = [item for item in items if item.type == type]

        if difficulty:
            items = [item for item in items if item.difficulty_level == difficulty]

        # Sort by category and order
        items.sort(key=lambda x: (x.category.value, x.order, x.title))

        # Apply pagination
        total = len(items)
        offset = (page - 1) * page_size
        items_page = items[offset : offset + page_size]

        # Generate categories summary
        categories = []
        category_counts = {}
        for item in documentation_items.values():
            cat = item.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat in DocumentationCategory:
            categories.append(
                {
                    "id": cat.value,
                    "name": cat.value.replace("_", " ").title(),
                    "count": category_counts.get(cat.value, 0),
                }
            )

        logger.info(f"Listed {len(items_page)} documentation items for user {current_user.id}")

        return DocumentationListResponse(
            items=items_page,
            categories=categories,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.error(f"Failed to list documentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documentation",
        )


@router.get("/{item_id}", response_model=DocumentationItem)
async def get_documentation_item(item_id: str, current_user: User = Depends(get_current_user)):
    """Get specific documentation item"""
    try:
        item = documentation_items.get(item_id)

        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Documentation item not found"
            )

        # Increment view count (in production, would update database)
        item.views += 1

        logger.info(f"Retrieved documentation item {item_id} for user {current_user.id}")

        return item

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get documentation item {item_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documentation item",
        )


@router.post("/search", response_model=list[SearchResult])
async def search_documentation(
    search_request: SearchRequest, current_user: User = Depends(get_current_user)
):
    """Search documentation content"""
    try:
        query = search_request.query.lower()
        results = []

        for item in documentation_items.values():
            # Apply filters
            if search_request.category and item.category != search_request.category:
                continue
            if search_request.type and item.type != search_request.type:
                continue
            if (
                search_request.difficulty_level
                and item.difficulty_level != search_request.difficulty_level
            ):
                continue
            if search_request.tags:
                if not any(tag in item.tags for tag in search_request.tags):
                    continue

            # Calculate relevance
            relevance = 0.0
            matching_sections = []
            highlight_text = ""

            # Title match (highest weight)
            if query in item.title.lower():
                relevance += 10.0
                highlight_text = item.title

            # Content match
            content_lower = item.content.lower()
            if query in content_lower:
                relevance += 5.0
                # Find context around match
                match_idx = content_lower.find(query)
                start = max(0, match_idx - 50)
                end = min(len(item.content), match_idx + 50 + len(query))
                highlight_text = item.content[start:end].strip()
                if start > 0:
                    highlight_text = "..." + highlight_text
                if end < len(item.content):
                    highlight_text = highlight_text + "..."

            # Tag match
            for tag in item.tags:
                if query in tag.lower():
                    relevance += 3.0
                    break

            # Section match
            for section in item.sections:
                if query in section.get("title", "").lower():
                    relevance += 2.0
                    matching_sections.append(section["title"])

            if relevance > 0:
                results.append(
                    SearchResult(
                        item=item,
                        relevance_score=relevance,
                        matching_sections=matching_sections,
                        highlight_text=highlight_text or item.content[:100] + "...",
                    )
                )

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        results = results[:50]

        logger.info(
            f"Search for '{query}' returned {len(results)} results for user {current_user.id}"
        )

        return results

    except Exception as e:
        logger.error(f"Failed to search documentation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search documentation",
        )


@router.get("/categories/{category}", response_model=list[DocumentationItem])
async def get_documentation_by_category(
    category: DocumentationCategory, current_user: User = Depends(get_current_user)
):
    """Get all documentation items in a specific category"""
    try:
        items = [item for item in documentation_items.values() if item.category == category]

        # Sort by order and title
        items.sort(key=lambda x: (x.order, x.title))

        logger.info(f"Retrieved {len(items)} items for category {category.value}")

        return items

    except Exception as e:
        logger.error(f"Failed to get documentation for category {category}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve category documentation",
        )


@router.post("/context-help")
async def get_contextual_help(context: HelpContext, current_user: User = Depends(get_current_user)):
    """Get contextual help based on current page/action"""
    try:
        # Simple contextual help mapping (in production, would be more sophisticated)
        help_mapping = {
            "/dashboard": "getting-started-overview",
            "/dashboard/templates": "attack-techniques-overview",
            "/dashboard/reports": "reporting-guide",
            "/dashboard/sessions": "session-management-guide",
        }

        suggested_item_id = help_mapping.get(context.page_url)

        if suggested_item_id and suggested_item_id in documentation_items:
            suggested_item = documentation_items[suggested_item_id]
        else:
            # Default help
            suggested_item = documentation_items.get("getting-started-overview")

        # Also suggest related items based on context
        related_items = []
        if context.feature:
            # Find items with matching tags
            for item in documentation_items.values():
                if context.feature.lower() in [tag.lower() for tag in item.tags]:
                    related_items.append(item)

        response = {
            "suggested_item": suggested_item,
            "related_items": related_items[:3],  # Limit to 3 related items
            "quick_actions": [
                {
                    "label": "View Getting Started",
                    "action": "navigate",
                    "target": "/docs/getting-started-overview",
                },
                {"label": "Search Documentation", "action": "search", "target": "/docs/search"},
                {"label": "Watch Tutorials", "action": "navigate", "target": "/docs/tutorials"},
            ],
        }

        logger.info(f"Provided contextual help for {context.page_url}")

        return response

    except Exception as e:
        logger.error(f"Failed to get contextual help: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get contextual help",
        )


@router.get("/videos/", response_model=list[VideoTutorial])
async def list_video_tutorials(
    category: DocumentationCategory | None = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_user),
):
    """List video tutorials"""
    try:
        tutorials = list(video_tutorials.values())

        if category:
            tutorials = [tutorial for tutorial in tutorials if tutorial.category == category]

        # Sort by category and views
        tutorials.sort(key=lambda x: (x.category.value, -x.views, x.title))

        logger.info(f"Listed {len(tutorials)} video tutorials")

        return tutorials

    except Exception as e:
        logger.error(f"Failed to list video tutorials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve video tutorials",
        )


@router.get("/quick-start")
async def get_quick_start_guide(current_user: User = Depends(get_current_user)):
    """Get quick start guide with personalized recommendations"""
    try:
        # Build personalized quick start based on user progress/preferences
        quick_start = {
            "welcome_message": f"Welcome to Chimera, {current_user.username}!",
            "progress": {
                "api_keys_configured": False,  # Would check actual status
                "first_assessment_completed": False,
                "first_report_generated": False,
            },
            "next_steps": [
                {
                    "title": "Configure API Keys",
                    "description": "Set up your LLM provider API keys",
                    "action": "navigate",
                    "target": "/dashboard/api-keys",
                    "estimated_time": "2 minutes",
                    "priority": "high",
                },
                {
                    "title": "Run Your First Assessment",
                    "description": "Use a template to run your first security test",
                    "action": "navigate",
                    "target": "/dashboard/templates",
                    "estimated_time": "5 minutes",
                    "priority": "medium",
                },
                {
                    "title": "Review functional Guidelines",
                    "description": "Understand responsible testing practices",
                    "action": "navigate",
                    "target": "/docs/functional-guidelines",
                    "estimated_time": "8 minutes",
                    "priority": "high",
                },
            ],
            "featured_content": [
                documentation_items.get("getting-started-overview"),
                documentation_items.get("functional-guidelines"),
                documentation_items.get("attack-techniques-overview"),
            ],
        }

        return quick_start

    except Exception as e:
        logger.error(f"Failed to get quick start guide: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quick start guide",
        )
