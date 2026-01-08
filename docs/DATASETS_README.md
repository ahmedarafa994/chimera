# Data Integration - Jailbreak Datasets

This repository includes an integration for in-the-wild jailbreak datasets (CSV / JSONL). Key artifacts:

- CLI: `scripts/import_jailbreak_datasets.py` — import files from `imported_data/` into the DB
- API: `GET /api/v1/datasets`, `GET /api/v1/datasets/{id}/prompts`, `POST /api/v1/datasets/import`
- Migration: `backend-api/alembic/versions/20251230_1200_add_jailbreak_datasets_and_prompts.py`
- Docs: `docs/INTEGRATE_JAILBREAK_DATASETS.md`

Run the migration and then import a sample dataset, e.g.:

```bash
alembic upgrade head
python scripts/import_jailbreak_datasets.py --file "jailbreak_llms-main/data/prompts/jailbreak_prompts_2023_12_25.csv" --name "jailbreak_llms_2023"
```
