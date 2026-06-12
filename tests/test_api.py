from __future__ import annotations

from fastapi.testclient import TestClient

from taslow_email_extraction_agent.app import app
from taslow_email_extraction_agent.dependencies import get_workflow_services


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_email_extraction_endpoint(base_request, services):
    app.dependency_overrides[get_workflow_services] = lambda: services
    try:
        client = TestClient(app)
        response = client.post(
            "/email-extractions",
            json=base_request.model_dump(by_alias=True, mode="json"),
            headers={"x-correlation-id": "corr-1"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "tasks_ready"
        assert payload["taskCandidateCount"] == 1
    finally:
        app.dependency_overrides.clear()
