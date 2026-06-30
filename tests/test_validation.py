from __future__ import annotations

from datetime import UTC, datetime

from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.validation import validate_assignments
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskAssignment


async def test_validation_collapses_duplicate_business_task_with_different_source_ids():
    assignments = [
        _assignment(
            source_task_id="extracted-task-1",
            title="Update portfolio reporting deck",
            description="Update the portfolio reporting deck with latest milestone status.",
            confidence=0.91,
        ),
        _assignment(
            source_task_id="extracted-task-2",
            title="Update reporting deck",
            description="Update the portfolio reporting deck with the latest milestone status.",
            confidence=0.88,
        ),
    ]

    result = await validate_assignments(assignments, _settings())

    assert len(result) == 1
    assert result[0].source_task_id == "extracted-task-1"
    assert "deduped_duplicate_task" in result[0].evidence


async def test_validation_keeps_same_task_requested_for_distinct_due_dates():
    assignments = [
        _assignment(
            source_task_id="extracted-task-1",
            title="Run utilization analysis",
            description="Run the utilization analysis for the program review.",
            due_date=datetime(2026, 6, 19, 17, 0, tzinfo=UTC),
        ),
        _assignment(
            source_task_id="extracted-task-2",
            title="Run utilization analysis",
            description="Run the utilization analysis for the program review.",
            due_date=datetime(2026, 6, 25, 17, 0, tzinfo=UTC),
        ),
    ]

    result = await validate_assignments(assignments, _settings())

    assert len(result) == 2
    assert {task.due_date.date().isoformat() for task in result if task.due_date} == {
        "2026-06-19",
        "2026-06-25",
    }


async def test_validation_merges_duplicate_when_one_copy_has_due_date():
    assignments = [
        _assignment(
            source_task_id="extracted-task-1",
            title="Send process maps",
            description="Send the latest business process maps to the trainer.",
            due_date=None,
            confidence=0.88,
        ),
        _assignment(
            source_task_id="extracted-task-2",
            title="Send latest process maps",
            description="Send the latest business process maps to the external trainer.",
            due_date=datetime(2026, 6, 22, 17, 0, tzinfo=UTC),
            confidence=0.90,
        ),
    ]

    result = await validate_assignments(assignments, _settings())

    assert len(result) == 1
    assert result[0].due_date is not None
    assert result[0].due_date.date().isoformat() == "2026-06-22"


async def test_validation_hard_blocks_external_sender_as_assignee():
    request = _request(
        from_email="client@example.com",
        from_name="Client",
        mailbox="operations@taslow.com",
        to=[{"email": "jesse@taslow.com", "name": "Jesse"}],
    )
    assignments = [
        _assignment(
            source_task_id="extracted-task-1",
            title="Handle client request",
            description="Handle the client request.",
            assignee_email="client@example.com",
            assignee_name="Client",
        )
    ]

    result = await validate_assignments(assignments, _settings(), request)

    assert result == []


async def test_validation_allows_internal_sender_as_assignee():
    request = _request(
        from_email="jesse@taslow.com",
        from_name="Jesse",
        mailbox="operations@taslow.com",
        to=[{"email": "tessa@taslow.com", "name": "Tessa"}],
    )
    assignments = [
        _assignment(
            source_task_id="extracted-task-1",
            title="Review owner notes",
            description="Review the owner notes.",
            assignee_email="jesse@taslow.com",
            assignee_name="Jesse",
        )
    ]

    result = await validate_assignments(assignments, _settings(), request)

    assert len(result) == 1
    assert result[0].assignee_email == "jesse@taslow.com"


def _settings() -> Settings:
    return Settings(assignee_confidence_threshold=0.80)


def _request(
    *,
    from_email: str,
    from_name: str,
    mailbox: str,
    to: list[dict[str, str]],
) -> EmailExtractionRequest:
    return EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox=mailbox,
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Project request",
        bodyText="Can someone handle this request?",
        **{"from": {"email": from_email, "name": from_name}},
        to=to,
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )


def _assignment(
    *,
    source_task_id: str,
    title: str,
    description: str,
    due_date: datetime | None = None,
    confidence: float = 0.90,
    assignee_email: str = "jesse@taslow.com",
    assignee_name: str = "Jesse",
) -> ExtractedTaskAssignment:
    return ExtractedTaskAssignment(
        sourceTaskId=source_task_id,
        title=title,
        description=description,
        projectId="project-1",
        scopeId="scope-1",
        scopeConfidence=0.82,
        assigneeEmail=assignee_email,
        assigneeName=assignee_name,
        assigneeConfidence=0.91,
        dueDate=due_date,
        dueDateConfidence=0.84 if due_date else None,
        overallConfidence=confidence,
        evidence=["test"],
    )
