import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.store import PlanStore


@pytest.fixture()
def client(tmp_path):
    store = PlanStore(str(tmp_path / "plans.db"))
    return TestClient(create_app(store))


HOT_BATCH = {
    "batch_id": "nightly-2026-11-22",
    "limits": {
        "s1": {"max_rows": 20000, "max_pairs": 300000},
        "s2": {"max_rows": 20000, "max_pairs": 300000},
        "s3": {"max_rows": 20000, "max_pairs": 300000},
    },
    "keys": [
        {"key": "dev-42|camp-7", "impressions": 10000, "clicks": 40, "shard": "s1"},
        {"key": "dev-1|camp-1", "impressions": 100, "clicks": 20, "shard": "s1"},
        {"key": "dev-2|camp-3", "impressions": 300, "clicks": 10, "shard": "s2"},
    ],
}

# Two cold keys whose combined rows overflow s1 -> no salting can help.
INFEASIBLE_BATCH = {
    "batch_id": "nightly-2026-11-22-bad",
    "limits": {"s1": {"max_rows": 20000, "max_pairs": 300000}},
    "keys": [
        {"key": "a", "impressions": 12000, "clicks": 0, "shard": "s1"},
        {"key": "b", "impressions": 9000, "clicks": 0, "shard": "s1"},
    ],
}

FIXED_BATCH = {
    **INFEASIBLE_BATCH,
    "batch_id": "nightly-2026-11-22-fixed",
    "limits": {"s1": {"max_rows": 25000, "max_pairs": 300000}},
}


def test_publish_then_fetch_current_and_by_version(client):
    r = client.post("/plans", json=HOT_BATCH)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == 1
    assert body["batch_id"] == "nightly-2026-11-22"
    assert len(body["splits"]) == 1
    assert all(l["within_limits"] for l in body["shard_loads"])

    cur = client.get("/plans/current")
    assert cur.status_code == 200 and cur.json()["version"] == 1

    by_version = client.get("/plans/1")
    assert by_version.status_code == 200
    assert by_version.json() == cur.json()

    listing = client.get("/plans")
    assert [m["version"] for m in listing.json()] == [1]


def test_hot_key_rule_is_reproducible_by_row_number(client):
    body = client.post("/plans", json=HOT_BATCH).json()
    (rule,) = body["splits"]
    assert rule["key"] == "dev-42|camp-7"
    assert rule["replicated_side"] == "clicks"
    assert rule["replicated_rows"] == 40
    assert rule["bucket_count"] == 2
    b0, b1 = rule["buckets"]
    assert (b0["row_start"], b0["row_end"]) == (1, 5000)
    assert (b1["row_start"], b1["row_end"]) == (5001, 10000)
    assert b0["shard"] == "s1"  # first bucket stays on the original shard
    assert b1["shard"] == "s2"  # second spills over: s1 pair budget exhausted
    assert sum(b["pairs"] for b in rule["buckets"]) == 10000 * 40


def test_failed_submission_keeps_previous_version_then_retry_succeeds(client):
    assert client.post("/plans", json=HOT_BATCH).status_code == 201

    r = client.post("/plans", json=INFEASIBLE_BATCH)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "infeasible_plan"
    assert detail["current_version"] == 1

    # previous version is still the live one, fully queryable
    cur = client.get("/plans/current")
    assert cur.status_code == 200 and cur.json()["version"] == 1

    # retry with corrected limits publishes a new version
    r2 = client.post("/plans", json=FIXED_BATCH)
    assert r2.status_code == 201, r2.text
    assert r2.json()["version"] == 2
    assert client.get("/plans/current").json()["version"] == 2

    # history retained: version 1 still queryable
    assert client.get("/plans/1").status_code == 200
    assert [m["version"] for m in client.get("/plans").json()] == [2, 1]


def test_current_404_before_any_publish(client):
    assert client.get("/plans/current").status_code == 404
    assert client.get("/plans/1").status_code == 404
    assert client.get("/plans").json() == []


def test_validation_errors(client):
    bad = {**HOT_BATCH, "keys": HOT_BATCH["keys"] + [HOT_BATCH["keys"][0]]}
    assert client.post("/plans", json=bad).status_code == 422  # duplicate key

    neg = {
        "batch_id": "x",
        "limits": {"s1": {"max_rows": 10, "max_pairs": 10}},
        "keys": [{"key": "a", "impressions": -1, "clicks": 0, "shard": "s1"}],
    }
    assert client.post("/plans", json=neg).status_code == 422

    missing_limit = {
        "batch_id": "x",
        "limits": {"s1": {"max_rows": 10, "max_pairs": 10}},
        "keys": [{"key": "a", "impressions": 1, "clicks": 0, "shard": "s9"}],
    }
    r = client.post("/plans", json=missing_limit)
    assert r.status_code == 422 and "s9" in r.json()["detail"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_default_store_created_at_startup(tmp_path, monkeypatch):
    # No store injected: lifespan must build one from PLAN_DB on startup.
    monkeypatch.setenv("PLAN_DB", str(tmp_path / "boot.db"))
    with TestClient(create_app()) as c:
        assert c.post("/plans", json=HOT_BATCH).status_code == 201
        assert c.get("/plans/current").json()["version"] == 1
    assert (tmp_path / "boot.db").exists()
