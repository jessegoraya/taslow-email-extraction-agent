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
    ExtractedTaskAssignment,
    ExtractedTaskCandidate,
    ExtractionStatus,
    Participant,
    ProjectContext,
    ProjectScope,
)
from taslow_email_extraction_agent.workflow import (
    _apply_ordered_multi_task_recipient,
    _match_explicit_scope_reference,
    _normalize_assignees_for_task,
    _retrieve_projects,
    _rewrite_ordered_multi_task_assignments,
    run_email_extraction,
)


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
    assert response.task_candidate_count == 0
    assert response.tasks == []
    assert response.diagnostics.stopped_after == "ProjectScoringExecutor"
    assert "task_candidates_discarded_no_project_match" in response.diagnostics.warnings


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


async def test_participant_candidates_expand_beyond_semantic_search(
    base_request, services, project
):
    semantic_only = ProjectContext(
        projectId="semantic-only",
        projectName="Education Program Support",
        associatedPeople=[AssociatedPerson(name="Morgan", email="morgan@tenant.com")],
    )
    services.project_client = InMemoryProjectClient([semantic_only, project])
    services.project_search_client = SemanticOnlyProjectSearchClient()

    projects = await _retrieve_projects(base_request, services, "draft status package")

    assert [candidate.project_id for candidate in projects] == ["semantic-only", "project-1"]
    assert projects[0].candidate_sources == ["azure_ai_search"]
    assert projects[1].candidate_sources == ["participant_overlap"]


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


async def test_explicit_scope_title_overrides_competing_semantic_result(
    base_request,
    services,
    project,
):
    project.scopes = [
        ProjectScope(
            scopeId="scope-explicit",
            title="Scope/General Description",
            description="General data analysis services.",
            groupTaskSetId="gts-explicit",
        ),
        ProjectScope(
            scopeId="scope-semantic",
            title="Work Requirements",
            description="Requirements summaries and task order deliverables.",
            groupTaskSetId="gts-semantic",
        ),
    ]
    services.project_search_client = CompetingScopeSearchClient()
    base_request.subject = "Requirements summary"
    base_request.body_text = (
        "Tessa, draft the requirements summary focused on "
        "Scope/General Description by next Friday at 5."
    )

    response = await run_email_extraction(base_request, services)

    assert response.status == ExtractionStatus.TASKS_READY
    assert response.tasks[0].scope_id == "scope-explicit"
    assert "explicit_scope_title_reference" in response.tasks[0].evidence


def test_explicit_scope_reference_ignores_quoted_message(base_request, project):
    project.scopes = [
        ProjectScope(
            scopeId="scope-current",
            title="Current Work",
            description="Current work.",
        ),
        ProjectScope(
            scopeId="scope-quoted",
            title="Quoted Work",
            description="Quoted work.",
        ),
    ]
    base_request.body_text = (
        "Tessa, please handle the current request.\n\n"
        "-----Original Message-----\n"
        "Please use Quoted Work."
    )

    assert _match_explicit_scope_reference(base_request, project) is None


def test_explicit_scope_reference_uses_protected_forwarded_context(
    base_request,
    project,
):
    scope_title = (
        "Failure To Meet The Above Transition-Out Requirements May Result In "
        "Withholding Of"
    )
    project.scopes = [
        ProjectScope(
            scopeId="scope-forwarded",
            title=scope_title,
            description="Transition-out payment requirements.",
        ),
        ProjectScope(
            scopeId="scope-other",
            title="Transition-Out Planning",
            description="General transition planning.",
        ),
    ]
    base_request.body_text = (
        f"> Tessa should document the {scope_title} analysis by 2026-08-17.\n\n"
        "The details above preserve the current conversation."
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Document the transition-out analysis",
        description="Document the transition-out analysis.",
        mentionedPeople=["Tessa"],
        dueText=None,
        confidence=0.93,
        evidence=["forwarded_actionable_context"],
    )

    match = _match_explicit_scope_reference(base_request, project, task)

    assert match is not None
    assert match.scope_id == "scope-forwarded"


def test_explicit_scope_reference_fails_closed_when_longest_match_is_ambiguous(
    base_request,
    project,
):
    project.scopes = [
        ProjectScope(scopeId="scope-1", title="Data Review", description="First."),
        ProjectScope(scopeId="scope-2", title="Data-Review", description="Second."),
    ]
    base_request.body_text = "Tessa, complete the Data Review by Friday."

    assert _match_explicit_scope_reference(base_request, project) is None


def test_explicit_scope_reference_prefers_subject_over_longer_body_context(
    base_request,
    project,
):
    project.scopes = [
        ProjectScope(
            scopeId="scope-requested",
            title="Contractor Furnished Items And Services",
            description="Items and services supplied by the contractor.",
        ),
        ProjectScope(
            scopeId="scope-contrast",
            title="Government Furnished Property, Equipment, And Services",
            description="Property and services supplied by the government.",
        ),
    ]
    base_request.subject = "Contractor furnished items and services update"
    base_request.body_text = (
        "Please update the Contractor Furnished Items And Services analysis. "
        "Explain what the contractor provides versus what is listed as "
        "Government Furnished Property, Equipment, And Services."
    )

    match = _match_explicit_scope_reference(base_request, project)

    assert match is not None
    assert match.scope_id == "scope-requested"


def test_explicit_scope_reference_prefers_body_when_subject_scope_is_project_name(
    base_request,
    project,
):
    project.project_name = "Fair Lending and Fair Housing Legal Advisory Services"
    project.scopes = [
        ProjectScope(
            scopeId="scope-project-name",
            title="Fair Lending And Fair Housing Legal Advisory Services",
            description="General project introduction.",
        ),
        ProjectScope(
            scopeId="scope-transition",
            title=(
                "Failure To Meet The Above Transition-Out Requirements May Result In Withholding Of"
            ),
            description="Transition-out payment requirements.",
        ),
    ]
    base_request.subject = (
        "Fair Lending and Fair Housing Legal Advisory Services - transition-out items"
    )
    base_request.body_text = (
        "Please update the Failure To Meet The Above Transition-Out Requirements "
        "May Result In Withholding Of analysis by 2026-08-12."
    )

    match = _match_explicit_scope_reference(base_request, project)

    assert match is not None
    assert match.scope_id == "scope-transition"


def test_explicit_child_scope_beats_project_name_repeated_in_body(
    base_request,
    project,
):
    project.project_name = "Fair Lending and Fair Housing Legal Advisory Services"
    project.scopes = [
        ProjectScope(
            scopeId="scope-project-name",
            title="Fair Lending And Fair Housing Legal Advisory Services",
            description="General project introduction.",
        ),
        ProjectScope(
            scopeId="scope-future-deliverables",
            title="Future Deliverables",
            description="Future task-order deliverables.",
        ),
    ]
    base_request.subject = "Fair Lending and Fair Housing Legal Advisory Services"
    base_request.body_text = (
        "Jesse, please reconcile the Future Deliverables analysis by 2026-08-19 "
        "for the Fair Lending and Fair Housing Legal Advisory Services project."
    )

    match = _match_explicit_scope_reference(base_request, project)

    assert match is not None
    assert match.scope_id == "scope-future-deliverables"


def test_explicit_scope_reference_prefers_protected_forwarded_scope_over_project_name(
    base_request,
    project,
):
    project.project_name = "Fair Lending and Fair Housing Legal Advisory Services"
    project.scopes = [
        ProjectScope(
            scopeId="scope-project-name",
            title="Fair Lending And Fair Housing Legal Advisory Services",
            description="General project introduction.",
        ),
        ProjectScope(
            scopeId="scope-transition",
            title=(
                "Failure To Meet The Above Transition-Out Requirements May Result In Withholding Of"
            ),
            description="Transition-out payment requirements.",
        ),
    ]
    base_request.subject = (
        "Fwd: Fair Lending and Fair Housing Legal Advisory Services"
    )
    base_request.body_text = (
        "From: External Program Lead\n"
        "Subject: Fair Lending and Fair Housing Legal Advisory Services\n\n"
        "Tessa should document the Failure To Meet The Above Transition-Out "
        "Requirements May Result In Withholding Of analysis by 2026-08-17."
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Document the transition-out analysis",
        description="Document the transition-out analysis.",
        mentionedPeople=["Tessa"],
        dueText=None,
        confidence=0.93,
        evidence=["forwarded_actionable_context"],
    )

    match = _match_explicit_scope_reference(base_request, project, task)

    assert match is not None
    assert match.scope_id == "scope-transition"


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


def test_ordered_multi_task_recipient_pairs_task_to_to_order(base_request, project):
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-2",
        title="Stage patient kiosk units",
        description="Stage patient kiosk units and verify the check-in firmware build.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    base_request.to = [
        Participant(email="tessa@tenant.com", name="Tessa"),
        Participant(email="jesse@tenant.com", name="Jesse"),
    ]
    assignees = [
        (
            project.people[0],
            0.85,
            ["recipient_overlap"],
        )
    ]

    selected = _apply_ordered_multi_task_recipient(
        base_request,
        project,
        task,
        assignees,
        task_index=1,
        task_count=2,
    )

    assert len(selected) == 1
    assert selected[0][0].email == "jesse@tenant.com"
    assert "ordered_multi_task_recipient_assignment" in selected[0][2]


def test_ordered_multi_task_assignment_rewrite_fixes_reused_fallback_assignee(
    base_request, project
):
    base_request.to = [
        Participant(email="tessa@tenant.com", name="Tessa"),
        Participant(email="jesse@tenant.com", name="Jesse"),
    ]
    assignments = [
        _assignment("task-1", "Task one", "tessa@tenant.com", "Tessa"),
        _assignment("task-2", "Task two", "tessa@tenant.com", "Tessa"),
    ]

    rewritten = _rewrite_ordered_multi_task_assignments(
        base_request,
        project,
        assignments,
        task_count=2,
    )

    assert [assignment.assignee_email for assignment in rewritten] == [
        "tessa@tenant.com",
        "jesse@tenant.com",
    ]
    assert "ordered_multi_task_recipient_assignment" in rewritten[1].evidence


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


class SemanticOnlyProjectSearchClient(FakeProjectSearchClient):
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                project_id="semantic-only",
                scope_id=None,
                score=0.94,
                rank=1,
                score_raw=0.94,
                score_margin=0.2,
            )
        ]


class CapturingScopeSearchClient(FakeProjectSearchClient):
    def __init__(self) -> None:
        self.scope_queries: list[str] = []

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        self.scope_queries.append(query_text)
        return await super().search_scopes(tenant_id, project_id, query_text)


class CompetingScopeSearchClient(FakeProjectSearchClient):
    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                project_id=project_id,
                scope_id="scope-semantic",
                score=0.94,
                rank=1,
                score_raw=0.94,
                score_margin=0.30,
            ),
            SearchCandidate(
                project_id=project_id,
                scope_id="scope-explicit",
                score=0.60,
                rank=2,
                score_raw=0.60,
                score_margin=0.0,
            ),
        ]


def _assignment(
    source_task_id: str,
    title: str,
    assignee_email: str,
    assignee_name: str,
) -> ExtractedTaskAssignment:
    return ExtractedTaskAssignment(
        sourceTaskId=source_task_id,
        title=title,
        description=title,
        projectId="project-1",
        scopeId="scope-1",
        scopeConfidence=0.90,
        assigneeEmail=assignee_email,
        assigneeName=assignee_name,
        assigneeConfidence=0.85,
        dueDate=None,
        dueDateConfidence=None,
        overallConfidence=0.85,
        evidence=["recipient_overlap", "multi_task_assignee_normalized"],
        needsReview=False,
    )
