from __future__ import annotations

from taslow_email_extraction_agent.clients.project_client import InMemoryProjectClient
from taslow_email_extraction_agent.clients.project_search_client import (
    ProjectSearchUnavailable,
    SearchCandidate,
)
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractionStatus
from taslow_email_extraction_agent.workflow import run_email_extraction


async def test_extracts_project_task(base_request, services):
    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert response.task_candidate_count == 1
    assert response.project_match is not None
    assert response.project_match.project_id == "project-1"
    assert len(response.tasks) == 1
    assert response.tasks[0].assignee_email == "tessa@tenant.com"
    assert response.tasks[0].scope_id == "scope-1"
    assert response.tasks[0].due_date is not None
    assert response.tasks[0].due_date.isoformat().startswith("2026-05-22T17:00:00")


async def test_no_task_short_circuits(base_request: EmailExtractionRequest, services):
    base_request.body_text = "Thanks for the update. This is helpful."
    base_request.subject = "FYI"

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.NO_TASK_FOUND
    assert response.tasks == []
    assert response.diagnostics.stopped_after == "TaskDetectionExecutor"


async def test_low_project_confidence_stops_before_write(base_request, services):
    services.project_client = InMemoryProjectClient([])

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.NO_PROJECT_MATCH
    assert response.tasks == []
    assert response.diagnostics.stopped_after == "ProjectScoringExecutor"


async def test_azure_search_candidates_are_hydrated_from_project_service(base_request, services):
    services.project_search_client = FakeProjectSearchClient()

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert response.project_match is not None
    assert response.project_match.project_id == "project-1"
    assert response.tasks[0].scope_id == "scope-1"
    assert "azure_ai_search_project_similarity" in response.project_match.evidence
    assert "azure_ai_search_scope_similarity" in response.tasks[0].evidence


async def test_azure_search_failure_returns_retryable(base_request, services):
    services.project_search_client = FailingProjectSearchClient()

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.RETRYABLE
    assert response.tasks == []
    assert response.diagnostics.retry_schedule == ["PT10M", "PT4H", "PT24H"]
    assert "azure_ai_search_dependency_failure" in response.diagnostics.warnings


class FakeProjectSearchClient:
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        return [SearchCandidate(project_id="project-1", scope_id=None, score=0.92)]

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        return [SearchCandidate(project_id=project_id, scope_id="scope-1", score=0.88)]


class FailingProjectSearchClient:
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        raise ProjectSearchUnavailable("boom")

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        raise ProjectSearchUnavailable("boom")
