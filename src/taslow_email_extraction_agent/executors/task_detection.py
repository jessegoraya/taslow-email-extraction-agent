from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from azure.core.exceptions import ClientAuthenticationError

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.azure_openai_auth import AzureOpenAIRequestAuthenticator
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskCandidate
from taslow_email_extraction_agent.text_utils import (
    has_forwarded_actionable_handoff,
    split_newest_and_quoted_text,
    token_set,
)

TASK_VERBS = [
    "please",
    "can you",
    "could you",
    "would you",
    "need you",
    "needs to",
    "must",
    "should",
    "update",
    "send",
    "create",
    "prepare",
    "review",
    "schedule",
    "set up",
    "follow up",
    "provide",
    "complete",
    "draft",
    "document",
    "incorporate",
    "can someone",
    "could someone",
    "would someone",
    "someone needs to",
    "does anyone know",
    "could anyone tell me",
    "is there anyone who can",
    "would anyone be willing to",
    "is it possible for someone to",
    "may i ask someone to",
    "would it be possible for anyone to",
    "open item",
    "remaining gap",
    "still outstanding",
    "yours to drive",
    "at risk of slipping",
]

GENERIC_REQUEST_RE = re.compile(
    r"\b(?:can\s+someone|could\s+someone|would\s+someone|someone\s+needs\s+to|"
    r"does\s+anyone\s+know|could\s+anyone\s+tell\s+me|"
    r"is\s+there\s+anyone\s+who\s+can|would\s+anyone\s+be\s+willing\s+to|"
    r"is\s+it\s+possible\s+for\s+someone\s+to|may\s+i\s+ask\s+someone\s+to|"
    r"would\s+it\s+be\s+possible\s+for\s+anyone\s+to)\b",
    re.IGNORECASE,
)

ACTIONABLE_OUTCOME_RE = re.compile(
    r"\b(?:send|update|confirm|schedule|review|complete|deliver|follow\s+up|"
    r"provide|prepare|draft|document|incorporate|create|revise|reconcile|resolve|"
    r"check|validate|summari[sz]e|capture|finalize|assess|evaluate|compare|"
    r"clarify|identify|flag|tighten|return|"
    r"verify|investigate|analy[sz]e|clean\s+up|coordinate|brief|upload|"
    r"attach|share|route|submit|approve|close|fix|tell\s+me|let\s+me\s+know|"
    r"answer|get\b.{0,40}\bdone|have\b.{0,80}\bready)\b",
    re.IGNORECASE,
)

DIRECT_REQUEST_RE = re.compile(
    r"\b(?:please|can\s+you|could\s+you|would\s+you|need\s+you\s+to|"
    r"i\s+need|we\s+need|i(?:'d|\s+would)\s+like|i\s+want\s+you\s+to|"
    r"i(?:'m|\s+am)\s+asking\s+you\s+to|if\s+you\s+can|"
    r"can\s+someone|could\s+someone|would\s+someone|someone\s+needs\s+to)\b",
    re.IGNORECASE,
)

UNRESOLVED_WORK_RE = re.compile(
    r"\b(?:open\s+item|remaining\s+gap|still\s+outstanding|remains?\s+outstanding|"
    r"overdue|at\s+risk\s+of\s+slipping|needs?\s+(?:attention|resolution|an?\s+update)|"
    r"needs?\s+to\s+be\s+(?:updated|refreshed|reconciled|resolved|completed|"
    r"reviewed|prepared|submitted|closed|fixed))\b",
    re.IGNORECASE,
)

CONTEXTUAL_UNRESOLVED_RE = re.compile(
    r"\b(?:pending|not\s+yet|has\s+not\s+been|have\s+not\s+been)\b",
    re.IGNORECASE,
)

COMMAND_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:[A-Z][a-z]+,\s*)?(?:please\s+)?"
    r"(?:send|update|confirm|schedule|review|complete|deliver|follow\s+up|"
    r"provide|prepare|draft|document|incorporate|create|revise|reconcile|resolve|"
    r"check|validate|summari[sz]e|capture|finalize|assess|evaluate|compare|"
    r"clarify|identify|flag|tighten|return|"
    r"verify|investigate|analy[sz]e|clean\s+up|coordinate|brief|upload|"
    r"attach|share|route|submit|approve|close|fix)\b"
)

COMPLETED_WORK_RE = re.compile(
    r"\b(?:already\s+(?:handled|completed|resolved|closed|done)|"
    r"(?:has|have)\s+been\s+(?:completed|resolved|closed|reconciled|updated|"
    r"refreshed|prepared|reviewed|validated|submitted)|"
    r"(?:is|are)\s+(?:complete|completed|resolved|closed|done)|"
    r"completed\s+(?:the|a|an|my|our)\b|"
    r"no\s+(?:further\s+)?action\s+(?:is\s+)?(?:needed|required)|"
    r"work\s+is\s+already\s+handled|cancel(?:led|ed)|withdrawn|retracted)\b",
    re.IGNORECASE,
)

STATUS_ONLY_RE = re.compile(
    r"\b(?:fyi|for\s+your\s+information|simple\s+check-in|status\s+(?:note|update)|"
    r"sharing\s+(?:a|the)\s+(?:brief\s+)?(?:update|summary)|"
    r"keeping\s+this\s+as\s+a\s+(?:simple\s+)?check-in)\b",
    re.IGNORECASE,
)

NO_ACTION_RE = re.compile(
    r"\b(?:no|nothing)\s+(?:further\s+|additional\s+|new\s+)?"
    r"(?:action|follow-up|follow\s+up|next\s+steps?|work|response)\s+"
    r"(?:is\s+)?(?:needed|required|requested)|"
    r"\b(?:i(?:'m|\s+am)\s+not\s+asking|not\s+asking)\s+for\s+anything\b",
    re.IGNORECASE,
)

COURTESY_CLOSING_RE = re.compile(
    r"\b(?:please\s+)?let\s+me\s+know\s+if\s+you\s+"
    r"(?:have\s+(?:any\s+)?questions|want\s+me\s+to|need\s+anything)|"
    r"\bfeel\s+free\s+to\s+(?:reach\s+out|ask)\b",
    re.IGNORECASE,
)

CONDITIONAL_NON_REQUEST_RE = re.compile(
    r"\bif\b.{0,100}\b(?:i|we|you)\s+(?:will\s+)?need\b",
    re.IGNORECASE,
)

FORWARDED_DELIVERY_WRAPPER_RE = re.compile(
    r"(?is)^\s*(?:"
    r"-{2,}\s*(?:original|forwarded)\s+message\s*-{2,}|"
    r"begin\s+forwarded\s+message:?|"
    r"forwarded\s+(?:message|note):?|"
    r">\s*from:|"
    r">\s*forwarded\s+(?:message|note):?|"
    r"from:\s+\S"
    r")"
)

DELIVERABLE_REQUEST_RE = re.compile(
    r"\b(?:analysis|assessment|brief|comments?|deliverable|document|draft|edits?|"
    r"evidence|findings?|matrix|notes?|package|plan|readout|report|response|"
    r"review|summary|tracker|update|write-?up)\b.{0,100}"
    r"\b(?:back|completed|done|due|finalized|ready|returned|submitted)\b"
    r".{0,40}\b(?:by|before|today|tomorrow|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

ACTION_TOKEN_RE = re.compile(
    r"\b(?:analy[sz]e|approve|assess|capture|check|clarify|close|compare|complete|"
    r"confirm|coordinate|create|deliver|document|draft|evaluate|finalize|fix|"
    r"identify|investigate|prepare|provide|reconcile|resolve|return|review|"
    r"schedule|send|share|submit|summari[sz]e|update|validate|verify)\b",
    re.IGNORECASE,
)

MENTION_RE = re.compile(r"(?<!\w)@?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
DUE_RE = re.compile(
    r"\b(?:by|before|due|tomorrow|today|next\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b[^.;,\n]*",
    re.IGNORECASE,
)


class TaskExtractor(Protocol):
    last_run_info: TaskExtractionRunInfo | None

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        """Extract task candidates from the email."""


@dataclass(slots=True)
class TaskExtractionRunInfo:
    provider: str
    model_deployment: str | None = None
    fallback_used: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    schema_valid: bool = True
    warning: str | None = None
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    recovery_reason: str | None = None


class HeuristicTaskExtractor:
    """Safe deterministic extractor used until the Foundry-backed extractor is configured."""

    def __init__(self) -> None:
        self.last_run_info: TaskExtractionRunInfo | None = None

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        self.last_run_info = TaskExtractionRunInfo(provider="heuristic", fallback_used=False)
        text = request.body_text.strip() or request.subject.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        candidates: list[ExtractedTaskCandidate] = []
        for sentence in sentences:
            lower = sentence.lower()
            if lower.startswith(("thanks", "thank you", "fyi")):
                continue
            if not any(verb in lower for verb in TASK_VERBS):
                continue
            if GENERIC_REQUEST_RE.search(sentence) and not ACTIONABLE_OUTCOME_RE.search(sentence):
                continue

            title = self._title_from_sentence(sentence)
            due_match = DUE_RE.search(sentence)
            mentioned = sorted({m.group(1).strip() for m in MENTION_RE.finditer(sentence)})
            candidates.append(
                ExtractedTaskCandidate(
                    sourceTaskId=f"extracted-task-{len(candidates) + 1}",
                    title=title,
                    description=sentence,
                    mentionedPeople=mentioned,
                    dueText=due_match.group(0).strip() if due_match else None,
                    confidence=0.74 if "please" in lower or "can you" in lower else 0.68,
                    evidence=["explicit_task_language"],
                )
            )
        return candidates

    @staticmethod
    def _title_from_sentence(sentence: str) -> str:
        cleaned = re.sub(
            r"^(please|can you|could you|would you|would you mind)\s+", "", sentence, flags=re.I
        )
        cleaned = cleaned.strip(" .?!")
        if len(cleaned) <= 80:
            return cleaned[0].upper() + cleaned[1:] if cleaned else "Email task"
        return cleaned[:77].rstrip() + "..."


class FoundryTaskExtractor:
    """Azure OpenAI / Foundry-backed task extractor with deterministic fallback."""

    def __init__(
        self,
        settings: Settings,
        fallback: TaskExtractor | None = None,
        authenticator: AzureOpenAIRequestAuthenticator | None = None,
    ) -> None:
        self._settings = settings
        self._fallback = fallback or HeuristicTaskExtractor()
        self._authenticator = authenticator or AzureOpenAIRequestAuthenticator(settings)
        self.last_run_info: TaskExtractionRunInfo | None = None

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        if not self._is_configured:
            return await self._fallback_with_info(request, "model_not_configured")

        try:
            candidates, input_tokens, output_tokens = await self._extract_with_model(request)
            recovery_reason = _task_recovery_reason(request) if not candidates else None
            recovery_attempted = recovery_reason is not None
            recovery_succeeded = False
            warning = None
            if recovery_reason:
                try:
                    recovered, recovery_input_tokens, recovery_output_tokens = (
                        await self._extract_with_model(
                            request,
                            system_prompt=_RECOVERY_SYSTEM_PROMPT,
                            recovery_reason=recovery_reason,
                        )
                    )
                    input_tokens = _sum_optional_tokens(input_tokens, recovery_input_tokens)
                    output_tokens = _sum_optional_tokens(output_tokens, recovery_output_tokens)
                    candidates = [
                        candidate.model_copy(
                            update={
                                "evidence": list(
                                    dict.fromkeys(
                                        [
                                            *candidate.evidence,
                                            "task_detection_recovery",
                                            recovery_reason,
                                        ]
                                    )
                                )
                            }
                        )
                        for candidate in recovered
                    ]
                    recovery_succeeded = bool(candidates)
                except (
                    ClientAuthenticationError,
                    httpx.HTTPError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as recovery_error:
                    warning = (
                        "task_detection_recovery_failed:"
                        f"{type(recovery_error).__name__}"
                    )
            self.last_run_info = TaskExtractionRunInfo(
                provider=self._settings.agent_task_extractor_provider,
                model_deployment=self._settings.azure_ai_model_deployment_name,
                fallback_used=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                schema_valid=True,
                warning=warning,
                recovery_attempted=recovery_attempted,
                recovery_succeeded=recovery_succeeded,
                recovery_reason=recovery_reason,
            )
            return candidates
        except (
            ClientAuthenticationError,
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            if not self._settings.agent_task_extractor_fallback_enabled:
                raise
            return await self._fallback_with_info(
                request, f"model_extraction_failed:{type(exc).__name__}"
            )

    @property
    def _is_configured(self) -> bool:
        return self._authenticator.is_configured

    async def _extract_with_model(
        self,
        request: EmailExtractionRequest,
        system_prompt: str | None = None,
        recovery_reason: str | None = None,
    ) -> tuple[list[ExtractedTaskCandidate], int | None, int | None]:
        endpoint = self._settings.azure_openai_endpoint
        assert endpoint is not None
        deployment = self._settings.azure_ai_model_deployment_name
        assert deployment is not None
        url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self._settings.azure_openai_chat_api_version}"
        )
        headers = await self._authenticator.get_headers()
        headers["Content-Type"] = "application/json"
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        _request_prompt_payload(request, recovery_reason),
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "taslow_task_extraction",
                    "strict": True,
                    "schema": _TASK_EXTRACTION_SCHEMA,
                },
            },
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()

        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        tasks = parsed.get("tasks", [])
        candidates = [
            ExtractedTaskCandidate.model_validate(
                {
                    "sourceTaskId": task.get("sourceTaskId") or f"extracted-task-{index}",
                    "title": task.get("title") or "Email task",
                    "description": task.get("description") or task.get("title") or "Email task",
                    "mentionedPeople": task.get("mentionedPeople") or [],
                    "dueText": task.get("dueText"),
                    "confidence": task.get("confidence") or 0.0,
                    "evidence": task.get("evidence") or [],
                }
            )
            for index, task in enumerate(tasks, start=1)
        ]
        usage = body.get("usage") or {}
        return candidates, usage.get("prompt_tokens"), usage.get("completion_tokens")

    async def _fallback_with_info(
        self, request: EmailExtractionRequest, warning: str
    ) -> list[ExtractedTaskCandidate]:
        candidates = await self._fallback.extract_tasks(request)
        fallback_info = getattr(self._fallback, "last_run_info", None)
        self.last_run_info = TaskExtractionRunInfo(
            provider=self._settings.agent_task_extractor_provider,
            model_deployment=self._settings.azure_ai_model_deployment_name,
            fallback_used=True,
            input_tokens=fallback_info.input_tokens if fallback_info else None,
            output_tokens=fallback_info.output_tokens if fallback_info else None,
            schema_valid=not warning.startswith("model_extraction_failed"),
            warning=warning,
        )
        return candidates


@step(name="TaskDetectionExecutor")
async def detect_tasks(
    request: EmailExtractionRequest,
    extractor: TaskExtractor,
) -> list[ExtractedTaskCandidate]:
    return _merge_overlapping_tasks(await extractor.extract_tasks(request))


_SYSTEM_PROMPT = """You extract actionable Taslow project tasks from corporate email.
Return only JSON matching the schema. Analyze only the newest authored message content.
Do not create tasks from stale quoted or forwarded content unless the newest message explicitly asks
the recipient to act on it. Return an empty tasks array for informational updates, meeting
logistics, status-only updates, approvals without a requested action, cancellation/retraction
language, or work that is already handled.
For forwarded actionable handoffs, the newest authored message controls whether work is assigned
and who is being asked to act, while the forwarded/original message may supply the task details,
Project context, Scope context, due date, and client rationale. Example: if the newest note says
"Jesse, can you handle this request below?", create the task from the forwarded client request and
preserve that context in the task description; do not assign based on the original client's
generic "can someone" wording.
When forwardedDeliveryActionRequest is true, the transport sender forwarded a concrete direct
request as the complete current message. Treat that delivery as a current handoff to the transport
recipients. Use the forwarded request for task, Project, Scope, and due-date context, but do not
assign work to the fictional external sender.
Implicit business-risk language can still be a task even without "please" or "can you" wording.
Create a task when the newest message clearly implies work is needed, such as:
- a log, tracker, dashboard, access queue, or record has not been updated and should be corrected;
- overlapping bookings, schedules, or calendar windows will cause a conflict if not resolved;
- someone wants or requested a write-up, summary, retrospective, deck, evidence package, or updated
  document;
- a stale queue, stuck approval, missing evidence, or outdated charter/process document needs to be
  cleared, refreshed, or resolved.
- phrases such as "open item", "remaining gap", "still outstanding", "this is yours to drive",
  or "at risk of slipping" identify unresolved work with an accountable actor or concrete outcome.
Create one task per distinct requested business outcome, not one task per sentence, clause, noun,
or supporting detail. If several clauses all support the same outcome, return one task with the
combined context. Extract multiple tasks only when the email clearly assigns separate outcomes that
could be completed independently by different owners or at different times.
Do not invent projects, assignees, due dates, or facts not present in the email."""

_RECOVERY_SYSTEM_PROMPT = """You are the conservative second-pass task detector for Taslow.
The first pass returned no tasks. Reconsider the email only because a deterministic runtime guard
found a strong action signal in the newest authored content. Return a task only when that content
directly requests, assigns, or identifies unresolved work with a concrete business outcome.
An implicit request can be actionable without the words "please" or "can you", but a Project
description, status report, completed analysis, current-state summary, future contract requirement,
or quoted prior instruction is not a new task.
Treat completed, reconciled, refreshed, resolved, closed, cancelled, retracted, FYI-only, and
"no action needed" content as non-actionable unless the newest authored text contains a separate
clear request for additional work.
For a forwarded handoff, use forwarded content only when the newest authored text explicitly asks
the recipient to handle the request below. The newest authored text controls current assignment.
When forwardedDeliveryActionRequest is true, the entire current delivery is a forwarded direct
request. It may be reconsidered as a current handoff only because the deterministic guard verified
concrete request language; status-only, completed, conditional, or FYI forwards remain non-tasks.
Return only JSON matching the schema. Do not invent Projects, Scopes, assignees, due dates, or
facts not present in the email."""


def _request_prompt_payload(
    request: EmailExtractionRequest,
    recovery_reason: str | None = None,
) -> dict:
    newest_body, quoted_context = split_newest_and_quoted_text(request.body_text)
    explicit_handoff = has_forwarded_actionable_handoff(request.body_text)
    forwarded_delivery_context = _forwarded_delivery_action_context(request)
    handoff = explicit_handoff or bool(forwarded_delivery_context)
    forwarded_context = quoted_context if explicit_handoff else forwarded_delivery_context
    payload = {
        "subject": request.subject,
        "newestAuthoredText": newest_body,
        "forwardedContextText": forwarded_context if handoff else "",
        "forwardedActionableHandoff": handoff,
        "forwardedDeliveryActionRequest": bool(forwarded_delivery_context),
        "forwardedHandoffPolicy": (
            "If forwardedDeliveryActionRequest is true, the transport sender's current delivery "
            "is the handoff and the transport recipients are the current assignment context. "
            "Otherwise, when forwardedActionableHandoff is true, extract the work item from the "
            "forwarded context but use the newest authored message to determine assignment intent."
        ),
        "sentDateTime": request.sent_date_time.isoformat() if request.sent_date_time else None,
        "direction": request.direction,
        "from": request.from_participant.model_dump() if request.from_participant else None,
        "to": [recipient.model_dump() for recipient in request.to],
        "cc": [recipient.model_dump() for recipient in request.cc],
        "conversationId": request.conversation_id,
        "parentMessageId": request.parent_message_id,
    }
    if recovery_reason:
        payload["taskDetectionRecoveryReason"] = recovery_reason
    return payload


def _task_recovery_reason(request: EmailExtractionRequest) -> str | None:
    if _forwarded_delivery_action_context(request):
        return "forwarded_delivery_action_request"

    newest_body, _quoted_context = split_newest_and_quoted_text(request.body_text)
    authored_text = newest_body.strip() or request.subject.strip()
    if not authored_text:
        return None

    if has_forwarded_actionable_handoff(request.body_text):
        return "forwarded_actionable_handoff"

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", authored_text)
        if sentence.strip()
    ]
    for sentence in sentences:
        if (
            NO_ACTION_RE.search(sentence)
            or COURTESY_CLOSING_RE.search(sentence)
            or CONDITIONAL_NON_REQUEST_RE.search(sentence)
        ):
            continue
        has_outcome = bool(ACTIONABLE_OUTCOME_RE.search(sentence))
        if DIRECT_REQUEST_RE.search(sentence) and has_outcome:
            return "direct_action_request"
        if COMMAND_RE.search(sentence) and not COMPLETED_WORK_RE.search(sentence):
            return "imperative_action_request"
        if (
            UNRESOLVED_WORK_RE.search(sentence)
            and not COMPLETED_WORK_RE.search(sentence)
            and not STATUS_ONLY_RE.search(authored_text)
        ):
            return "unresolved_work_signal"
        if (
            DELIVERABLE_REQUEST_RE.search(sentence)
            and not COMPLETED_WORK_RE.search(sentence)
            and not STATUS_ONLY_RE.search(authored_text)
        ):
            return "deliverable_deadline_request"
        if (
            CONTEXTUAL_UNRESOLVED_RE.search(sentence)
            and has_outcome
            and not COMPLETED_WORK_RE.search(sentence)
            and not STATUS_ONLY_RE.search(authored_text)
        ):
            return "unresolved_work_signal"

    if COMPLETED_WORK_RE.search(authored_text) or STATUS_ONLY_RE.search(authored_text):
        return None
    return None


def _forwarded_delivery_action_context(request: EmailExtractionRequest) -> str:
    body = request.body_text.strip()
    if not body or not FORWARDED_DELIVERY_WRAPPER_RE.search(body):
        return ""

    for sentence in _action_sentences(body):
        if (
            NO_ACTION_RE.search(sentence)
            or COURTESY_CLOSING_RE.search(sentence)
            or CONDITIONAL_NON_REQUEST_RE.search(sentence)
            or COMPLETED_WORK_RE.search(sentence)
            or STATUS_ONLY_RE.search(sentence)
        ):
            continue
        if (
            DIRECT_REQUEST_RE.search(sentence)
            and ACTIONABLE_OUTCOME_RE.search(sentence)
        ) or (
            COMMAND_RE.search(sentence)
            and not COMPLETED_WORK_RE.search(sentence)
        ):
            return body
    return ""


def _action_sentences(value: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", value)
        if sentence.strip()
    ]


def _sum_optional_tokens(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


_TASK_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sourceTaskId": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "mentionedPeople": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "dueText": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "sourceTaskId",
                    "title",
                    "description",
                    "mentionedPeople",
                    "dueText",
                    "confidence",
                    "evidence",
                ],
            },
        }
    },
    "required": ["tasks"],
}


def _merge_overlapping_tasks(
    candidates: list[ExtractedTaskCandidate],
) -> list[ExtractedTaskCandidate]:
    merged: list[ExtractedTaskCandidate] = []
    for candidate in candidates:
        target_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _should_merge_tasks(existing, candidate)
            ),
            None,
        )
        if target_index is None:
            merged.append(candidate)
            continue
        merged[target_index] = _merge_task_pair(merged[target_index], candidate)
    return [
        task.model_copy(update={"source_task_id": f"extracted-task-{index}"})
        for index, task in enumerate(merged, start=1)
    ]


def _should_merge_tasks(left: ExtractedTaskCandidate, right: ExtractedTaskCandidate) -> bool:
    if _explicit_due_dates_conflict(left.due_text, right.due_text):
        return False

    left_tokens = token_set(" ".join([left.title, left.description]))
    right_tokens = token_set(" ".join([right.title, right.description]))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    left_actions = {match.lower() for match in ACTION_TOKEN_RE.findall(
        " ".join([left.title, left.description])
    )}
    right_actions = {match.lower() for match in ACTION_TOKEN_RE.findall(
        " ".join([right.title, right.description])
    )}
    same_actions = bool(left_actions) and left_actions == right_actions
    left_description = " ".join(left.description.lower().split())
    right_description = " ".join(right.description.lower().split())
    contained = (
        left_description in right_description or right_description in left_description
    )
    exact_title = " ".join(left.title.lower().split()) == " ".join(
        right.title.lower().split()
    )
    return exact_title or (same_actions and (contained or overlap >= 0.70))


def _explicit_due_dates_conflict(left: str | None, right: str | None) -> bool:
    left_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", left or ""))
    right_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", right or ""))
    return bool(left_dates and right_dates and left_dates.isdisjoint(right_dates))


def _merge_task_pair(
    left: ExtractedTaskCandidate,
    right: ExtractedTaskCandidate,
) -> ExtractedTaskCandidate:
    description = (
        left.description if len(left.description) >= len(right.description) else right.description
    )
    title = left.title if len(left.title) <= len(right.title) else right.title
    evidence = list(
        dict.fromkeys([*left.evidence, *right.evidence, "merged_overlapping_task_candidates"])
    )
    return left.model_copy(
        update={
            "title": title,
            "description": description,
            "mentioned_people": sorted(set(left.mentioned_people) | set(right.mentioned_people)),
            "due_text": left.due_text or right.due_text,
            "confidence": max(left.confidence, right.confidence),
            "evidence": evidence,
        }
    )
