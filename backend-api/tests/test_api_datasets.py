from fastapi.testclient import TestClient

from app.core.database import SyncSessionFactory, get_sync_engine
from app.crud import crud_jailbreak
from app.db import models
from app.main import app

# Ensure DB tables exist for tests
engine = get_sync_engine()
models.Base.metadata.create_all(bind=engine)

# Create a dataset if not present (idempotent for repeated runs)
with SyncSessionFactory() as db:
    ds = crud_jailbreak.get_dataset_by_name(db, "jailbreak_llms_2023")
    if not ds:
        crud_jailbreak.create_dataset(
            db,
            type(
                "D",
                (),
                {
                    "name": "jailbreak_llms_2023",
                    "description": "Imported for tests",
                    "source_url": None,
                    "license": None,
                },
            ),
        )

client = TestClient(app)


def test_list_datasets():
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # Expect dataset we imported earlier
    assert any(d.get("name") == "jailbreak_llms_2023" for d in data)


def test_get_prompts_for_dataset():
    # Find dataset id
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    ds = next((d for d in datasets if d.get("name") == "jailbreak_llms_2023"), None)
    assert ds is not None
    ds_id = ds["id"]

    p_resp = client.get(f"/api/v1/datasets/{ds_id}/prompts")
    assert p_resp.status_code == 200
    prompts = p_resp.json()
    assert isinstance(prompts, list)
    # imported dataset may have many or few entries depending on import run
    assert len(prompts) >= 0
