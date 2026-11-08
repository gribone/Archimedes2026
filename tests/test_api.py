"""
Starlette TestClient verification of Archimedes2026 API routes:
1. GET /api/health -> HTTP 200 {"status": "ok"}
2. GET /api/bootstrap -> HTTP 200 with schema validation
3. GET /api/company/1 -> HTTP 200 via test fixture/mock
4. POST /api/run -> HTTP 200 with calculated payload and generation
"""

from __future__ import annotations

from typing import Any
import pytest
from starlette.testclient import TestClient

from src.server import app, get_service


# -------------------------------------------------------------------------
# Test Fixtures for API
# -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate Starlette TestClient against app."""
    return TestClient(app)


@pytest.fixture
def mock_company_1(monkeypatch):
    """
    Mock fixture providing deterministic return for company_id = 1.
    Prevents external network verification calls and supplies valid company payload.
    """
    service = get_service()
    orig_company = service.company

    def fake_company(company_id: int) -> dict[str, Any] | None:
        if company_id == 1:
            return {
                "company": {
                    "company_id": 1,
                    "vat": "012345678",
                    "company_name_el": "Δοκιμαστική Επιχείρηση 1 ΙΚΕ",
                    "name": "Δοκιμαστική Επιχείρηση 1 ΙΚΕ",
                    "legal_type": "Ι.Κ.Ε.",
                    "city": "Αθήνα",
                    "prefecture_name": "ΑΤΤΙΚΗΣ",
                    "municipality_name": "ΑΘΗΝΑΙΩΝ",
                    "primary_kad": "56101100",
                    "primary_kad_desc": "Υπηρεσίες παροχής γευμάτων",
                    "incorporation_date": "2020-01-15",
                    "latitude": 37.9838,
                    "longitude": 23.7275,
                },
                "kads": [
                    {
                        "activity_type": "Κύρια",
                        "activity_id": "56101100",
                        "description": "Υπηρεσίες παροχής γευμάτων",
                        "date_from": "2020-01-15",
                    }
                ],
                "context": {
                    "is_excluded": False,
                    "verification": {
                        "operational_status": "active",
                        "confidence": 0.95,
                        "reasons": ["Tax active", "Commercial registry entry"],
                    },
                },
            }
        return orig_company(company_id)

    monkeypatch.setattr(service, "company", fake_company)
    return fake_company


# -------------------------------------------------------------------------
# 1. Health Check Tests
# -------------------------------------------------------------------------
@pytest.mark.api
def test_api_health(client: TestClient):
    """Verify GET /api/health responds with HTTP 200 and status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    data = resp.json()
    assert data == {"status": "ok"}


@pytest.mark.api
def test_api_health_legacy_endpoint(client: TestClient):
    """Verify legacy /_stcore/health endpoint remains functional."""
    resp = client.get("/_stcore/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# -------------------------------------------------------------------------
# 2. Bootstrap Contract Tests
# -------------------------------------------------------------------------
@pytest.mark.api
def test_api_bootstrap_schema(client: TestClient):
    """Verify GET /api/bootstrap responds with HTTP 200 and required fields."""
    resp = client.get("/api/bootstrap")
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level keys
    expected_keys = {
        "defaults",
        "param_specs",
        "precision",
        "benchmark",
        "optimal_weights",
        "initial",
    }
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from bootstrap response"

    # Verify defaults schema
    defaults = data["defaults"]
    assert "min_icps" in defaults
    assert "max_icps" in defaults
    assert "sim_weight_feature_cosine" in defaults
    assert "sim_weight_eb_lift_affinity" in defaults
    assert "sim_weight_spatial_kernel" in defaults

    # Verify initial calculation payload
    initial = data["initial"]
    assert "profiles" in initial
    assert "lookalikes" in initial
    assert "clients" in initial or "assignments" in initial
    assert len(initial["profiles"]) >= 1


# -------------------------------------------------------------------------
# 3. Company Inspection Endpoint Tests (/api/company/{company_id})
# -------------------------------------------------------------------------
@pytest.mark.api
def test_api_company_1_with_mock_fixture(client: TestClient, mock_company_1):
    """Verify GET /api/company/1 responds with HTTP 200 when fixture-mocked."""
    resp = client.get("/api/company/1")
    assert resp.status_code == 200
    data = resp.json()

    assert "company" in data
    assert data["company"]["company_id"] == 1
    assert data["company"]["name"] == "Δοκιμαστική Επιχείρηση 1 ΙΚΕ"
    assert "kads" in data
    assert isinstance(data["kads"], list)
    assert len(data["kads"]) >= 1
    assert "context" in data


@pytest.mark.api
def test_api_company_real_dataset_id(client: TestClient, valid_dataset_company_id: int):
    """Verify GET /api/company/92460 succeeds for a real dataset company without mock."""
    resp = client.get(f"/api/company/{valid_dataset_company_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "company" in data
    assert data["company"]["company_id"] == valid_dataset_company_id


@pytest.mark.api
def test_api_company_not_found(client: TestClient):
    """Verify non-existent company ID returns HTTP 404."""
    resp = client.get("/api/company/999999999")
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("status") == "not_found"


@pytest.mark.api
def test_api_company_invalid_id_format(client: TestClient):
    """Verify alphanumeric or non-integer company ID returns 404 (route constraint)."""
    resp = client.get("/api/company/invalid_company_id")
    assert resp.status_code in (400, 404)


# -------------------------------------------------------------------------
# 4. Interactive Run Endpoint Tests (POST /api/run)
# -------------------------------------------------------------------------
@pytest.mark.api
@pytest.mark.slow
def test_api_run_fast_precision_returns_200(client: TestClient):
    """Verify POST /api/run with fast precision executes and returns HTTP 200."""
    payload = {
        "params": {
            "precision": "fast",
            "min_icps": 2,
            "max_icps": 2,
            "sim_weight_feature_cosine": 0.60,
            "sim_weight_eb_lift_affinity": 0.30,
            "sim_weight_spatial_kernel": 0.10,
        }
    }
    resp = client.post("/api/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data.get("status") == "ok"
    assert isinstance(data.get("generation"), int)
    assert "data" in data

    run_data = data["data"]
    assert "profiles" in run_data
    assert "lookalikes" in run_data
    assert "clients" in run_data or "assignments" in run_data


@pytest.mark.api
def test_api_run_validation_error_zero_weights(client: TestClient):
    """Verify POST /api/run with zero weights returns HTTP 400 Bad Request."""
    payload = {
        "params": {
            "sim_weight_feature_cosine": 0.0,
            "sim_weight_eb_lift_affinity": 0.0,
            "sim_weight_spatial_kernel": 0.0,
        }
    }
    resp = client.post("/api/run", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data.get("status") == "error"
    assert "βάρος" in data.get("message", "").lower() or "weight" in data.get("message", "").lower()


@pytest.mark.api
def test_api_run_empty_payload_uses_defaults(client: TestClient):
    """Verify POST /api/run with empty body {} succeeds using default parameters."""
    resp = client.post("/api/run", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


# -------------------------------------------------------------------------
# 5. Concurrency Safety Tests
# -------------------------------------------------------------------------
@pytest.mark.api
def test_api_concurrent_requests(client: TestClient):
    """Verify that concurrent requests to API endpoints execute safely without 500 errors."""
    import concurrent.futures

    def hit_endpoint(i: int) -> int:
        if i % 3 == 0:
            return client.get("/api/health").status_code
        elif i % 3 == 1:
            return client.get("/api/bootstrap").status_code
        else:
            return client.get("/api/company/92460").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        statuses = list(ex.map(hit_endpoint, range(24)))

    assert all(status == 200 for status in statuses)
