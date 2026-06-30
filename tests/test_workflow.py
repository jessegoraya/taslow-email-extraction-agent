from __future__ import annotations

from taslow_email_extraction_agent.clients.project_client import InMemoryProjectClient
from taslow_email_extraction_agent.clients.project_search_client import (
    ProjectSearchUnavailable,
    SearchCandidate,
)
from taslow_email_extraction_agent.clients.task_history_client import InMemoryTaskHistoryClient
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ExtractionStatus,
    Participant,
)
from taslow_email_extraction_agent.workflow import _normalize_assignees_for_task, run_email_extraction


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
    assert response.diagnostics.project_hydration_provider == "project-agent-context"
    assert response.diagnostics.search_query_count == 1
    assert response.diagnostics.scope_search_query_count == 1
    assert response.diagnostics.project_scoring is not None
    assert response.diagnostics.project_scoring.search_score_normalized == 0.92
    assert response.diagnostics.project_scoring.search_rank == 1
    assert response.diagnostics.project_candidate_scores
    assert response.diagnostics.project_candidate_scores[0].project_id == "project-1"


async def test_azure_search_failure_returns_retryable(base_request, services):
    services.project_search_client = FailingProjectSearchClient()

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.RETRYABLE
    assert response.tasks == []
    assert response.diagnostics.retry_schedule == ["PT10M", "PT4H", "PT24H"]
    assert "azure_ai_search_dependency_failure" in response.diagnostics.warnings


async def test_scope_search_uses_newest_authored_task_text(base_request, services):
    search_client = CapturingScopeSearchClient()
    services.project_search_client = search_client
    base_request.body_text = (
        "Tessa, please update the electrical notes before the review.\n\n"
        "-----Original Message-----\n"
        "Subject: Lodging coordination\n"
        "Please book rooms for next week."
    )

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert search_client.scope_queries
    assert "Lodging coordination" not in search_client.scope_queries[0]
    assert "book rooms" not in search_client.scope_queries[0]


async def test_forwarded_handoff_uses_forwarded_context_for_scope_search(base_request, services):
    search_client = CapturingScopeSearchClient()
    services.project_search_client = search_client
    base_request.body_text = (
        "Tessa, can you handle this request below?\n\n"
        "-----Original Message-----\n"
        "From: Client <client@example.com>\n"
        "Subject: Electrical notes\n"
        "Can someone update the electrical notes before the review?"
    )

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert response.tasks[0].assignee_email == "tessa@tenant.com"
    assert search_client.scope_queries
    assert any("electrical notes" in query.lower() for query in search_client.scope_queries)


async def test_direct_assignment_language_overrides_single_recipient(base_request, services):
    base_request.body_text = "Jesse, have Tessa update the electrical scope by next Friday at 5."
    base_request.to = [Participant(email="jesse@tenant.com", name="Jesse")]

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert response.tasks[0].assignee_email == "tessa@tenant.com"
    assert "delegated_assignment_language" in response.tasks[0].evidence


async def test_invite_context_person_is_not_extra_assignee(base_request, services):
    base_request.body_text = "Jesse, can you add Tessa to Friday's review invite?"
    base_request.to = [Participant(email="jesse@tenant.com", name="Jesse")]

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert len(response.tasks) == 1
    assert response.tasks[0].assignee_email == "jesse@tenant.com"
    assert "direct_address_assignment" in response.tasks[0].evidence


async def test_thread_context_is_recorded_for_successful_extraction(base_request, services):
    base_request.conversation_id = "conversation-1"
    services.task_history_client = InMemoryTaskHistoryClient()

    response = await run_email_extraction(base_request, services)
    thread_context = await services.task_history_client.get_thread_context(base_request)

    assert response.status == ExtractionStatus.TASKS_READY
    assert thread_context is not None
    assert thread_context.project_id == "project-1"


def test_multi_task_assignee_normalization_prevents_cross_product():
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Update Devon's warranty claim process",
        description="Devon should update the warranty claim process.",
        mentionedPeople=["Devon"],
        dueText=None,
        confidence=0.90,
        evidence=["implicit_task_language"],
    )
    devon = AssociatedPerson(name="Devon Price", email="devon@acme-consulting.example")
    nina = AssociatedPerson(name="Nina Patel", email="nina@acme-consulting.example")
    normalized = _normalize_assignees_for_task(
        task,
        [
            (devon, 0.92, ["named_request_actor_signal"]),
            (nina, 0.91, ["recipient_overlap"]),
        ],
        task_count=4,
    )

    assert len(normalized) == 1
    assert normalized[0][0].email == "devon@acme-consulting.example"
    assert "multi_task_assignee_normalized" in normalized[0][2]


def test_multi_task_assignee_normalization_uses_unused_fallback_recipient():
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-2",
        title="Update action item two",
        description="Update action item two.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    victor = AssociatedPerson(name="Victor Chen", email="victor@acme-consulting.example")
    sara = AssociatedPerson(name="Sara Ahmed", email="sara@acme-consulting.example")
    normalized = _normalize_assignees_for_task(
        task,
        [
            (victor, 0.85, ["recipient_overlap"]),
            (sara, 0.85, ["recipient_overlap"]),
        ],
        task_count=2,
        used_assignee_emails={"victor@acme-consulting.example"},
    )

    assert len(normalized) == 1
    assert normalized[0][0].email == "sara@acme-consulting.example"
    assert "multi_task_assignee_normalized" in normalized[0][2]


class FakeProjectSearchClient:
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                project_id="project-1",
                scope_id=None,
                score=0.92,
                rank=1,
                score_raw=0.92,
                score_margin=0.2,
            )
        ]

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                project_id=project_id,
                scope_id="scope-1",
                score=0.88,
                rank=1,
                score_raw=0.88,
                score_margin=0.1,
            )
        ]


class FailingProjectSearchClient:
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        raise ProjectSearchUnavailable("boom")

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        raise ProjectSearchUnavailable("boom")


class CapturingScopeSearchClient(FakeProjectSearchClient):
    def __init__(self) -> None:
        self.scope_queries: list[str] = []

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        self.scope_queries.append(query_text)
        return await super().search_scopes(tenant_id, project_id, query_text)
