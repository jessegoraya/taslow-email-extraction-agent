from __future__ import annotations

from typing import Any

from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.task_detection import (
    FoundryTaskExtractor,
    _merge_overlapping_tasks,
    _request_prompt_payload,
    _task_recovery_reason,
)
from taslow_email_extraction_agent.models import (
    EmailExtractionRequest,
    ExtractedTaskCandidate,
)


class _ConfiguredAuthenticator:
    is_configured = True


class _SequencedFoundryTaskExtractor(FoundryTaskExtractor):
    def __init__(self, responses: list[Any]) -> None:
        super().__init__(
            Settings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_ai_model_deployment_name="test-model",
                agent_task_extractor_fallback_enabled=False,
            ),
            authenticator=_ConfiguredAuthenticator(),  # type: ignore[arg-type]
        )
        self._responses = list(responses)
        self.calls: list[tuple[str | None, str | None]] = []

    async def _extract_with_model(
        self,
        request: EmailExtractionRequest,
        system_prompt: str | None = None,
        recovery_reason: str | None = None,
    ) -> tuple[list[ExtractedTaskCandidate], int | None, int | None]:
        self.calls.append((system_prompt, recovery_reason))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_recovery_guard_detects_direct_action_request(base_request):
    base_request.body_text = (
        "Please update the Operational Service Delivery analysis by August 18. "
        "Include any language that should be clarified."
    )

    assert _task_recovery_reason(base_request) == "direct_action_request"


def test_recovery_guard_detects_unresolved_work(base_request):
    base_request.body_text = (
        "The access review remains outstanding and needs to be completed "
        "before the release window."
    )

    assert _task_recovery_reason(base_request) == "unresolved_work_signal"


def test_recovery_guard_detects_imperative_action(base_request):
    base_request.body_text = "Reconcile the transition report with the approved staffing plan."

    assert _task_recovery_reason(base_request) == "imperative_action_request"


def test_recovery_guard_detects_forwarded_actionable_handoff(base_request):
    base_request.body_text = (
        "Tessa, can you handle the request below?\n\n"
        "-----Original Message-----\n"
        "Please update the electrical scope before Friday."
    )

    assert _task_recovery_reason(base_request) == "forwarded_actionable_handoff"


def test_recovery_guard_detects_forwarded_delivery_action_request(base_request):
    base_request.body_text = (
        "Begin forwarded message:\n"
        "From: External Program Lead\n"
        "To: tessa@tenant.example\n"
        "Please document the C-14 deliverable analysis by Friday."
    )

    assert _task_recovery_reason(base_request) == "forwarded_delivery_action_request"


def test_recovery_guard_rejects_forwarded_delivery_status(base_request):
    base_request.body_text = (
        "Begin forwarded message:\n"
        "From: External Program Lead\n"
        "To: tessa@tenant.example\n"
        "The analysis has been completed and no further action is needed."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_completed_work(base_request):
    base_request.body_text = (
        "I have completed the operational service delivery analysis. "
        "The draft has been reconciled and no further action is needed."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_status_only_update(base_request):
    base_request.body_text = (
        "This is a quick status update. The analysis is moving forward and "
        "the current draft remains consistent with the source material."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_pending_status_without_action(base_request):
    base_request.body_text = (
        "Status update: the client decision remains pending. "
        "This is for awareness only."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_have_context_status_language(base_request):
    base_request.body_text = (
        "I wanted to share the update so you have the latest context. "
        "Nothing additional is needed from you at this point."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_courtesy_closing(base_request):
    base_request.body_text = (
        "This is an informational update only. "
        "Please let me know if you want me to add anything to the meeting notes."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_conditional_future_need(base_request):
    base_request.body_text = (
        "Nothing new is needed from your side right now. "
        "I will send another note if we need to compare updates."
    )

    assert _task_recovery_reason(base_request) is None


def test_recovery_guard_rejects_quoted_only_request(base_request):
    base_request.body_text = (
        "FYI only.\n\n"
        "-----Original Message-----\n"
        "Please update the electrical scope before Friday."
    )

    assert _task_recovery_reason(base_request) is None


def test_prompt_payload_uses_normalized_message_blocks(base_request):
    base_request.body_text = (
        "FYI only.\n\n"
        "-----Original Message-----\n"
        "Please update the electrical scope before Friday."
    )

    payload = _request_prompt_payload(base_request)

    assert "bodyText" not in payload
    assert payload["newestAuthoredText"] == "FYI only."
    assert payload["forwardedContextText"] == ""
    assert payload["forwardedActionableHandoff"] is False


def test_prompt_payload_includes_forwarded_context_only_for_handoff(base_request):
    base_request.body_text = (
        "Tessa, can you handle this request below?\n\n"
        "-----Original Message-----\n"
        "Please update the electrical scope before Friday."
    )

    payload = _request_prompt_payload(base_request, "forwarded_actionable_handoff")

    assert payload["newestAuthoredText"] == "Tessa, can you handle this request below?"
    assert "update the electrical scope" in payload["forwardedContextText"]
    assert payload["forwardedActionableHandoff"] is True
    assert payload["forwardedDeliveryActionRequest"] is False
    assert payload["taskDetectionRecoveryReason"] == "forwarded_actionable_handoff"


def test_prompt_payload_includes_complete_forwarded_delivery(base_request):
    base_request.body_text = (
        "-----Original Message-----\n"
        "From: External Program Lead\n"
        "To: tessa@tenant.example\n"
        "Please reconcile the transition analysis by Friday."
    )

    payload = _request_prompt_payload(
        base_request,
        "forwarded_delivery_action_request",
    )

    assert payload["newestAuthoredText"] == ""
    assert "reconcile the transition analysis" in payload["forwardedContextText"]
    assert payload["forwardedActionableHandoff"] is True
    assert payload["forwardedDeliveryActionRequest"] is True


async def test_foundry_extractor_recovers_empty_first_pass(base_request):
    base_request.body_text = "Please update the electrical scope before Friday."
    recovered_task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Update the electrical scope",
        description="Update the electrical scope before Friday.",
        mentionedPeople=[],
        dueText="before Friday",
        confidence=0.93,
        evidence=["direct_request"],
    )
    extractor = _SequencedFoundryTaskExtractor(
        [
            ([], 10, 2),
            ([recovered_task], 7, 4),
        ]
    )

    tasks = await extractor.extract_tasks(base_request)

    assert len(tasks) == 1
    assert "task_detection_recovery" in tasks[0].evidence
    assert "direct_action_request" in tasks[0].evidence
    assert len(extractor.calls) == 2
    assert extractor.calls[1][1] == "direct_action_request"
    assert extractor.last_run_info is not None
    assert extractor.last_run_info.input_tokens == 17
    assert extractor.last_run_info.output_tokens == 6
    assert extractor.last_run_info.recovery_attempted is True
    assert extractor.last_run_info.recovery_succeeded is True
    assert extractor.last_run_info.recovery_reason == "direct_action_request"


async def test_foundry_extractor_does_not_reconsider_no_task_control(base_request):
    base_request.body_text = (
        "I completed the electrical scope review. The draft has been updated "
        "and no further action is needed."
    )
    extractor = _SequencedFoundryTaskExtractor(
        [
            ([], 10, 2),
        ]
    )

    tasks = await extractor.extract_tasks(base_request)

    assert tasks == []
    assert len(extractor.calls) == 1
    assert extractor.last_run_info is not None
    assert extractor.last_run_info.recovery_attempted is False
    assert extractor.last_run_info.recovery_succeeded is False
    assert extractor.last_run_info.recovery_reason is None


async def test_foundry_extractor_records_unsuccessful_recovery(base_request):
    base_request.body_text = "Please update the electrical scope before Friday."
    extractor = _SequencedFoundryTaskExtractor(
        [
            ([], 10, 2),
            ([], 7, 4),
        ]
    )

    tasks = await extractor.extract_tasks(base_request)

    assert tasks == []
    assert extractor.last_run_info is not None
    assert extractor.last_run_info.recovery_attempted is True
    assert extractor.last_run_info.recovery_succeeded is False
    assert extractor.last_run_info.recovery_reason == "direct_action_request"


def test_overlapping_tasks_with_distinct_explicit_due_dates_remain_separate():
    candidates = [
        ExtractedTaskCandidate(
            sourceTaskId="task-1",
            title="Analyze Government Furnished Property and Services",
            description="Analyze the Government Furnished Property and Services section.",
            mentionedPeople=[],
            dueText="2026-08-13",
            confidence=0.94,
            evidence=["direct_request"],
        ),
        ExtractedTaskCandidate(
            sourceTaskId="task-2",
            title="Summarize Government Furnished Property and Services",
            description="Summarize the Government Furnished Property and Services section.",
            mentionedPeople=["Jgoraya"],
            dueText="2026-08-14",
            confidence=0.94,
            evidence=["named_request"],
        ),
    ]

    merged = _merge_overlapping_tasks(candidates)

    assert len(merged) == 2
    assert [task.due_text for task in merged] == ["2026-08-13", "2026-08-14"]


def test_duplicate_tasks_with_same_due_date_still_merge():
    candidates = [
        ExtractedTaskCandidate(
            sourceTaskId="task-1",
            title="Review the Facility Security analysis",
            description="Review the Facility Security analysis before the next cycle.",
            mentionedPeople=["David"],
            dueText="2026-08-22",
            confidence=0.91,
            evidence=["direct_request"],
        ),
        ExtractedTaskCandidate(
            sourceTaskId="task-2",
            title="Review Facility Security analysis",
            description="Review the Facility Security analysis before the next cycle.",
            mentionedPeople=["David"],
            dueText="2026-08-22",
            confidence=0.93,
            evidence=["duplicate_model_candidate"],
        ),
    ]

    merged = _merge_overlapping_tasks(candidates)

    assert len(merged) == 1
    assert "merged_overlapping_task_candidates" in merged[0].evidence
