from __future__ import annotations

from taslow_email_extraction_agent.executors.due_date import normalize_due_date
from taslow_email_extraction_agent.models import ExtractedTaskCandidate


async def test_forwarded_actionable_context_recovers_iso_due_date(base_request):
    base_request.body_text = (
        "> Tessa should validate the Electrical Scope analysis by 2026-08-17.\n\n"
        "The details above preserve the current conversation."
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Validate the Electrical Scope analysis",
        description="Validate the Electrical Scope analysis.",
        mentionedPeople=["Tessa"],
        dueText=None,
        confidence=0.93,
        evidence=["forwarded_actionable_context"],
    )

    due_date, confidence, evidence = await normalize_due_date(base_request, task)

    assert due_date is not None
    assert due_date.isoformat().startswith("2026-08-17T17:00:00")
    assert confidence == 0.78
    assert evidence == ["explicit_due_date"]


async def test_stale_quoted_due_date_is_not_used_without_forwarded_marker(base_request):
    base_request.body_text = (
        "FYI only.\n\n"
        "> Tessa should validate the Electrical Scope analysis by 2026-08-17."
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Review the FYI",
        description="Review the informational note.",
        mentionedPeople=["Tessa"],
        dueText=None,
        confidence=0.93,
        evidence=[],
    )

    due_date, confidence, evidence = await normalize_due_date(base_request, task)

    assert due_date is None
    assert confidence is None
    assert evidence == []
