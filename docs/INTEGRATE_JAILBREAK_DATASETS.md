# Integrating External Jailbreak Datasets

This document explains how to import and manage external jailbreak datasets within Chimera.

## Overview
- New tables: `jailbreak_datasets`, `jailbreak_prompts`
- Service: `app.services.jailbreak_importer.JailbreakImporter` supports CSV and JSONL sources
- CLI script: `scripts/import_jailbreak_datasets.py`
- API endpoints: `GET /api/v1/datasets`, `GET /api/v1/datasets/{id}/prompts`, `POST /api/v1/datasets/import`

## Importing locally
Example:

```bash
python scripts/import_jailbreak_datasets.py --file "jailbreak_llms-main/data/prompts/jailbreak_prompts_2023_12_25.csv" --name "jailbreak_llms_2023"
```

## Notes
- Files are expected under `imported_data/` directory.
- The importer normalizes multiple schemas (jailbreak_llms, PAIR, etc.).
- For large imports, run in background or chunk with `--max-lines`.
- Run Alembic migration after reviewing and generating migration.
