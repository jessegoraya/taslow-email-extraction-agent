from __future__ import annotations

from taslow_email_extraction_agent.executors.reranking import (
    _generic_deliverable_override_blocked,
    _has_strong_deterministic_assignee,
    _scope_payload,
    _should_skip_scope_reranker,
)
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectScope,
)


def test_scope_payload_includes_work_outcome_policy_and_hints():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="FSM - duplicate vendor records showing up again",
        bodyText=(
            "Jesse, duplicate vendor records are appearing in the reconciliation output. "
            "Can someone dig into where the mismatched source data is coming from?"
        ),
        **{"from": {"email": "ramona@example.com", "name": "Ramona"}},
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Investigate duplicate vendor source data",
        description="Find and fix the mismatched source data causing duplicate vendor records.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["implicit_task_language"],
    )
    project = ProjectContext(
        projectId="project-1",
        projectName="FSM",
        description="Financial systems modernization",
        associatedPeople=[],
        associatedManagers=[],
        scopes=[
            ProjectScope(
                scopeId="scope-1",
                title="Data Management and Quality Improvement",
                description="Correct source data quality issues.",
            )
        ],
    )

    payload = _scope_payload(request, task, project, project.scopes[0], 0.7, [])

    assert "scopeSelectionPolicy" in payload
    assert "work outcome" in payload["scopeSelectionPolicy"]["primaryRule"]
    assert "data_quality_or_source_data" in payload["workOutcomeHints"]["hints"]
    assert "reconciliation_context_or_work" in payload["workOutcomeHints"]["hints"]


def test_assignee_reranker_skips_strong_deterministic_owner_evidence():
    assignees = [
        (
            AssociatedPerson(name="David Vance", email="david@tenant.com"),
            0.89,
            ["subject_matter_owner_signal"],
        )
    ]

    assert _has_strong_deterministic_assignee(assignees)


def test_assignee_reranker_does_not_skip_weak_deterministic_evidence():
    assignees = [
        (
            AssociatedPerson(name="Jesse", email="jesse@tenant.com"),
            0.88,
            ["single_visible_project_recipient"],
        )
    ]

    assert not _has_strong_deterministic_assignee(assignees)


def test_scope_reranker_skips_high_deterministic_confidence():
    scope = ProjectScope(
        scopeId="scope-1",
        title="Emergency Department Expansion",
        description="Construction coordination and phasing.",
        searchScore=0.68,
        searchRank=1,
        searchMargin=0.02,
    )

    assert _should_skip_scope_reranker(scope, 0.86)


def test_scope_reranker_skips_strong_rank_one_search_margin():
    scope = ProjectScope(
        scopeId="scope-1",
        title="Billing and Reconciliation",
        description="Rate audits and invoice reconciliation.",
        searchScore=0.74,
        searchRank=1,
        searchMargin=0.08,
    )

    assert _should_skip_scope_reranker(scope, 0.80)


def test_scope_reranker_runs_for_low_confidence_or_close_candidates():
    low_confidence_scope = ProjectScope(
        scopeId="scope-1",
        title="Billing and Reconciliation",
        description="Rate audits and invoice reconciliation.",
        searchScore=0.66,
        searchRank=1,
        searchMargin=0.09,
    )
    close_scope = ProjectScope(
        scopeId="scope-2",
        title="Travel Reservation Configuration",
        description="Reservation and ticketing configuration.",
        searchScore=0.69,
        searchRank=1,
        searchMargin=0.02,
    )

    assert not _should_skip_scope_reranker(low_confidence_scope, 0.76)
    assert not _should_skip_scope_reranker(close_scope, 0.80)


def test_generic_deliverable_override_requires_stronger_scope_evidence():
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Finalize architecture review report",
        description="Finalize the report for the architecture diagram review feedback.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["implicit_task_language"],
    )
    current_scope = ProjectScope(
        scopeId="scope-1",
        title="Architecture Engineering Services",
        description="Architecture review and engineering design feedback.",
    )
    selected_scope = ProjectScope(
        scopeId="scope-2",
        title="Reporting Decks and Submittals",
        description="Generic decks, reports, binders, and submittal packages.",
    )
    result = {
        "scopeId": "scope-2",
        "confidence": 0.84,
        "rationale": "The task asks for a report deliverable.",
    }

    assert _generic_deliverable_override_blocked(
        task,
        result,
        current_scope,
        selected_scope,
        0.82,
        0.84,
    )


def test_generic_deliverable_override_allows_strong_specific_evidence():
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Finalize audit report for billing reconciliation",
        description="Finalize the audit report for the invoice variance reconciliation.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["implicit_task_language"],
    )
    current_scope = ProjectScope(
        scopeId="scope-1",
        title="Lodging Coordination",
        description="Hotel room booking and logistics.",
    )
    selected_scope = ProjectScope(
        scopeId="scope-2",
        title="Billing and Reconciliation",
        description="Invoice variance audits and reconciliation.",
    )
    result = {
        "scopeId": "scope-2",
        "confidence": 0.93,
        "rationale": "Billing reconciliation and invoice variance are direct scope evidence.",
    }

    assert not _generic_deliverable_override_blocked(
        task,
        result,
        current_scope,
        selected_scope,
        0.82,
        0.93,
    )


def test_scope_payload_strips_quoted_text_and_includes_billing_rate_hint():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="TMC - hotel rate audit",
        bodyText=(
            "Jesse, the hotel rate audit found another CBA cycle billing variance. "
            "Could someone please reconcile the invoice before closeout?\n\n"
            "-----Original Message-----\n"
            "Subject: Lodging coordination\n"
            "Please book rooms for next week."
        ),
        **{"from": {"email": "client@example.com", "name": "Client"}},
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Reconcile the invoice billing variance",
        description="Reconcile the hotel rate audit billing variance before closeout.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["implicit_task_language"],
    )
    project = ProjectContext(
        projectId="project-1",
        projectName="TMC",
        description="Travel Management Center",
        associatedPeople=[],
        associatedManagers=[],
        scopes=[
            ProjectScope(
                scopeId="scope-1",
                title="Billing and Reconciliation",
                description="Rate audits, billing variance, and invoice reconciliation.",
            )
        ],
    )

    payload = _scope_payload(request, task, project, project.scopes[0], 0.7, [])

    assert "Lodging coordination" not in payload["email"]["bodyText"]
    assert "billing_reconciliation_or_rate_audit" in payload["workOutcomeHints"]["hints"]
    examples = payload["scopeSelectionPolicy"]["contextVsOutcomeExamples"]
    assert any("rate" in example["preferScopeAbout"] for example in examples)


def test_scope_payload_includes_forwarded_context_for_actionable_handoff():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Forwarded client request",
        bodyText=(
            "Jesse, can you handle this request below?\n\n"
            "-----Original Message-----\n"
            "From: Client <client@example.com>\n"
            "Subject: CBA rate audit\n"
            "Can someone reconcile the hotel rate variance before closeout?"
        ),
        **{"from": {"email": "manager@tenant.com", "name": "Manager"}},
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Handle the forwarded client request",
        description="Handle the forwarded client request.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["forwarded_actionable_handoff"],
    )
    project = ProjectContext(
        projectId="project-1",
        projectName="TMC",
        description="Travel Management Center",
        associatedPeople=[],
        associatedManagers=[],
        scopes=[
            ProjectScope(
                scopeId="scope-1",
                title="Billing and Reconciliation",
                description="Rate audits and invoice reconciliation.",
            )
        ],
    )

    payload = _scope_payload(request, task, project, project.scopes[0], 0.7, [])

    assert payload["email"]["forwardedActionableHandoff"] is True
    assert "hotel rate variance" in payload["email"]["forwardedContextText"]
    assert "hotel rate variance" in payload["task"]["contextText"]
    assert "forwarded_actionable_handoff_context" in payload["workOutcomeHints"]["hints"]


def test_scope_payload_includes_new_deterministic_outcome_hints():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="TMC reservation system - fare class codes",
        bodyText=(
            "Jesse, can Debra update the official travel reservation and ticketing "
            "configuration for the new fare class codes?"
        ),
        **{"from": {"email": "client@example.com", "name": "Client"}},
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Update fare class code configuration",
        description="Update the travel reservation and ticketing configuration.",
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["explicit_task_language"],
    )
    project = ProjectContext(
        projectId="project-1",
        projectName="TMC",
        description="Travel Management Center",
        associatedPeople=[],
        associatedManagers=[],
        scopes=[
            ProjectScope(
                scopeId="scope-1",
                title="Official Travel Reservation and Ticketing",
                description="Travel reservation configuration.",
            )
        ],
    )

    payload = _scope_payload(request, task, project, project.scopes[0], 0.7, [])

    assert "travel_reservation_ticketing_configuration" in payload["workOutcomeHints"]["hints"]
