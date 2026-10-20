"""
tests/unit/test_api.py
Unit tests for FastAPI endpoints: /health, /v1/sources, /v1/ask validation, /v1/check-claim.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexus.api.main import app
from nexus.api.models import AskRequest, ClaimRequest

client = TestClient(app)


@pytest.mark.unit
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data


@pytest.mark.unit
def test_sources_endpoint():
    response = client.get("/v1/sources")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "name" in first
    assert "tier" in first


@pytest.mark.unit
def test_ask_request_validation():
    # Query must not be empty
    req = AskRequest(query="What is the repo rate?", top_k=5)
    assert req.query == "What is the repo rate?"
    assert req.top_k == 5

    with pytest.raises(Exception):
        AskRequest(query="")


@pytest.mark.unit
def test_claim_request_validation():
    req = ClaimRequest(claim="RBI banned 500 rupee notes")
    assert req.claim == "RBI banned 500 rupee notes"

    with pytest.raises(Exception):
        ClaimRequest(claim="")


@pytest.mark.unit
def test_ingest_endpoint_auth():
    # Without valid API key, must return 401 Unauthorized
    response = client.post("/v1/ingest", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
