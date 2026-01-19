from app.domain.prompt_library_models import PromptTemplate, TemplateRating
from app.repositories.prompt_library_repository import prompt_library_repository
from app.schemas.prompt_library import RateTemplateRequest, RatingStatistics


class TemplateRatingService:
    async def rate_template(
        self, template_id: str, user_id: str, request: RateTemplateRequest
    ) -> PromptTemplate | None:
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        # Check if user already rated, if so update it
        existing_rating_idx = -1
        for i, r in enumerate(template.ratings):
            if r.user_id == user_id:
                existing_rating_idx = i
                break

        new_rating = TemplateRating(
            user_id=user_id,
            rating=request.rating,
            effectiveness_vote=request.effectiveness_vote,
            comment=request.comment,
        )

        if existing_rating_idx >= 0:
            template.ratings[existing_rating_idx] = new_rating
        else:
            template.ratings.append(new_rating)

        return await prompt_library_repository.update(template)

    async def get_ratings(self, template_id: str) -> list[TemplateRating]:
        """Get all ratings for a template"""
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return []
        return template.ratings

    async def get_rating_statistics(self, template_id: str) -> RatingStatistics:
        """Get rating statistics for a template"""
        template = await prompt_library_repository.get_by_id(template_id)

        if not template or not template.ratings:
            return RatingStatistics(
                avg_rating=0.0,
                total_ratings=0,
                effectiveness_score=0.0,
                rating_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            )

        # Calculate rating distribution
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in template.ratings:
            if 1 <= rating.rating <= 5:
                distribution[rating.rating] += 1

        return RatingStatistics(
            avg_rating=template.avg_rating,
            total_ratings=template.total_ratings,
            effectiveness_score=template.effectiveness_score,
            rating_distribution=distribution,
        )

    async def update_rating(
        self, template_id: str, user_id: str, request: RateTemplateRequest
    ) -> PromptTemplate | None:
        """Update an existing rating"""
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return None

        # Find existing rating
        for i, r in enumerate(template.ratings):
            if r.user_id == user_id:
                template.ratings[i] = TemplateRating(
                    user_id=user_id,
                    rating=request.rating,
                    effectiveness_vote=request.effectiveness_vote,
                    comment=request.comment,
                )
                return await prompt_library_repository.update(template)

        return None  # Rating not found

    async def delete_rating(self, template_id: str, user_id: str) -> bool:
        """Delete a user's rating for a template"""
        template = await prompt_library_repository.get_by_id(template_id)
        if not template:
            return False

        # Find and remove rating
        for i, r in enumerate(template.ratings):
            if r.user_id == user_id:
                template.ratings.pop(i)
                await prompt_library_repository.update(template)
                return True

        return False


# Singleton instance
template_rating_service = TemplateRatingService()
