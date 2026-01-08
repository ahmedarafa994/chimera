# AutoDAN-Turbo PDF Integration Plan

## Goal
Integrate the specific strategies and methodologies described in the AutoDAN-Turbo paper (arXiv:2410.05295v4) into the current codebase.

## Analysis
The current codebase (`pipeline.py`, `attacker_autodan_reasoning.py`) already implements the core AutoDAN-Turbo architecture (Lifelong Learning, Strategy Library, Summarizer). However, it lacks the specific "seed" strategies described in Appendix G of the paper, which are crucial for the "Warm-up" and "Human-Designed Strategy Injection" phases described in Section 3.4.

## Implementation Steps

- [x] **Step 1**: Create Strategy Data File
    - Extract the 16 strategies (15 from Appendix G + "Future Pacing" from Appendix H) into a structured JSON file: `backend-api/app/services/autodan/data/autodan_turbo_strategies.json`.
    - Ensure format matches the Strategy Library schema: `{"Strategy": "...", "Definition": "...", "Example": "..."}`.

- [x] **Step 2**: Update Data Loader
    - Modify `backend-api/app/services/autodan/jailbreak_data_loader.py` to include a method `get_autodan_turbo_strategies()` that loads this new JSON file.

- [x] **Step 3**: Verify Integration
    - Create a script `verify_pdf_integration.py` that:
        1. Loads the strategies.
        2. Initializes `AttackerAutoDANReasoning`.
        3. Calls `use_strategy` with one of the PDF strategies.
        4. Verifies the generated prompt contains the strategy logic.

## Success Criteria
- All strategies from the PDF are correctly formatted and loadable.
- The system can recognize and use these strategies for "Warm Start" as described in the paper.