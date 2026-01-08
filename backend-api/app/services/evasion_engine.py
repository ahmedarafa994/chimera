import logging
from typing import Any

from sqlalchemy.orm import Session

from app.crud.llm_crud import get_llm_model
from app.schemas.api_schemas import EvasionAttemptResult, EvasionTaskConfig, LLMProvider
from app.services.llm_factory import LLMFactory
from app.services.metamorphosis_strategies import STRATEGY_REGISTRY, BaseMetamorphosisStrategy

logger = logging.getLogger(__name__)


class EvasionEngine:
    """
    The core engine for orchestrating metamorphic evasion tasks.
    It applies a chain of metamorphosis strategies and interacts with target LLMs.
    """

    def __init__(self, db: Session):
        self.db = db

    async def execute_evasion_attempt(
        self, task_id: str, evasion_config: EvasionTaskConfig, attempt_number: int
    ) -> EvasionAttemptResult:
        """
        Executes a single evasion attempt: applies transformations and queries the LLM.
        """
        initial_prompt = evasion_config.initial_prompt
        transformed_prompt = initial_prompt
        transformation_log: list[dict[str, Any]] = []

        logger.info(
            f"Task {task_id}: Starting attempt {attempt_number} for model {evasion_config.target_model_id}"
        )

        # 1. Get Target LLM Adapter
        db_llm_model = get_llm_model(self.db, evasion_config.target_model_id)
        if not db_llm_model:
            raise ValueError(
                f"Target LLM model with ID {evasion_config.target_model_id} not found."
            )

        # In a real scenario, API keys should be retrieved securely, not from DB config directly.
        # For PoC, we'll simulate fetching an API key (it won't be in DB_LLMModel.config)
        # and pass it to the factory.
        # The 'api_key' argument to get_adapter should come from a secure source (e.g., K-V store)
        # For now, we'll pass placeholder from config if available.
        llm_api_key = db_llm_model.config.get(
            "api_key", ""
        )  # This should ideally be from an env var or secret manager

        llm_adapter = LLMFactory.get_adapter(
            model_id=db_llm_model.id,
            provider=LLMProvider(db_llm_model.provider),
            model_name=db_llm_model.name,
            api_key=llm_api_key,  # WARNING: insecure for production
            config=db_llm_model.config,
        )

        # 2. Apply Metamorphosis Strategies
        for strategy_config in evasion_config.strategy_chain:
            strategy_name = strategy_config.name
            strategy_params = strategy_config.params

            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if not strategy_class:
                raise ValueError(f"Unknown metamorphosis strategy: {strategy_name}")

            strategy_instance: BaseMetamorphosisStrategy = strategy_class(strategy_params)

            # Context for strategies, e.g., for SemanticRephrasing to use an LLM
            # For PoC, SemanticRephrasing is synchronous and has a placeholder.
            # A full implementation would make the `transform` method async if it needs LLM calls.
            # Or the engine could pre-process steps needing LLM calls.
            strategy_context = (
                {"llm_adapter": llm_adapter} if strategy_name == "SemanticRephrasing" else {}
            )

            previous_prompt = transformed_prompt
            try:
                transformed_prompt = strategy_instance.transform(
                    transformed_prompt, strategy_context
                )
                transformation_log.append(
                    {
                        "strategy": strategy_name,
                        "params": strategy_params,
                        "input_prompt": previous_prompt,
                        "output_prompt": transformed_prompt,
                        "status": "success",
                    }
                )
                logger.debug(
                    f"Task {task_id}: Strategy {strategy_name} applied. New prompt length: {len(transformed_prompt)}"
                )
            except Exception as e:
                transformation_log.append(
                    {
                        "strategy": strategy_name,
                        "params": strategy_params,
                        "input_prompt": previous_prompt,
                        "output_prompt": transformed_prompt,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                logger.error(f"Task {task_id}: Failed to apply strategy {strategy_name}: {e}")
                # Decide whether to stop the chain or continue with the last valid prompt
                # For now, we stop.
                raise

        # 3. Query Target LLM
        llm_response = ""
        try:
            llm_response = await llm_adapter.generate_response(
                transformed_prompt, **db_llm_model.config
            )
            logger.info(f"Task {task_id}: LLM responded. Response length: {len(llm_response)}")
        except Exception as e:
            logger.error(f"Task {task_id}: Error querying LLM {db_llm_model.name}: {e}")
            raise

        # 4. Evaluate Response
        evaluation_results: dict[str, Any] = await llm_adapter.evaluate_response(
            prompt=transformed_prompt,
            response=llm_response,
            criteria=evasion_config.success_criteria,
        )
        is_evasion_successful = evaluation_results.get("is_evasion_successful", False)
        logger.info(
            f"Task {task_id}: Evasion attempt {attempt_number} evaluation: {'SUCCESS' if is_evasion_successful else 'FAILURE'}"
        )

        return EvasionAttemptResult(
            attempt_number=attempt_number,
            transformed_prompt=transformed_prompt,
            llm_response=llm_response,
            is_evasion_successful=is_evasion_successful,
            evaluation_details=evaluation_results,
            transformation_log=transformation_log,
        )
