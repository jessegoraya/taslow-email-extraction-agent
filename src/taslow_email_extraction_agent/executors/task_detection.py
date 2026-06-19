from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskCandidate
from taslow_email_extraction_agent.text_utils import token_set

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
]

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

    def __init__(self, settings: Settings, fallback: TaskExtractor | None = None) -> None:
        self._settings = settings
        self._fallback = fallback or HeuristicTaskExtractor()
        self.last_run_info: TaskExtractionRunInfo | None = None

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        if not self._is_configured:
            return await self._fallback_with_info(request, "model_not_configured")

        try:
            candidates, input_tokens, output_tokens = await self._extract_with_model(request)
            self.last_run_info = TaskExtractionRunInfo(
                provider=self._settings.agent_task_extractor_provider,
                model_deployment=self._settings.azure_ai_model_deployment_name,
                fallback_used=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                schema_valid=True,
            )
            return candidates
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if not self._settings.agent_task_extractor_fallback_enabled:
                raise
            return await self._fallback_with_info(
                request, f"model_extraction_failed:{type(exc).__name__}"
            )

    @property
    def _is_configured(self) -> bool:
        return bool(
            self._settings.azure_openai_endpoint
            and self._settings.azure_openai_api_key
            and self._settings.azure_ai_model_deployment_name
        )

    async def _extract_with_model(
        self, request: EmailExtractionRequest
    ) -> tuple[list[ExtractedTaskCandidate], int | None, int | None]:
        endpoint = self._settings.azure_openai_endpoint
        assert endpoint is not None
        deployment = self._settings.azure_ai_model_deployment_name
        assert deployment is not None
        url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self._settings.azure_openai_chat_api_version}"
        )
        headers = {
            "api-key": self._settings.azure_openai_api_key or "",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(_request_prompt_payload(request), ensure_ascii=True),
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
Create one task per distinct requested business outcome, not one task per sentence, clause, noun,
or supporting detail. If several clauses all support the same outcome, return one task with the
combined context. Extract multiple tasks only when the email clearly assigns separate outcomes that
could be completed independently by different owners or at different times.
Do not invent projects, assignees, due dates, or facts not present in the email."""


def _request_prompt_payload(request: EmailExtractionRequest) -> dict:
    return {
        "subject": request.subject,
        "bodyText": request.body_text,
        "sentDateTime": request.sent_date_time.isoformat() if request.sent_date_time else None,
        "direction": request.direction,
        "from": request.from_participant.model_dump() if request.from_participant else None,
        "to": [recipient.model_dump() for recipient in request.to],
        "cc": [recipient.model_dump() for recipient in request.cc],
        "conversationId": request.conversation_id,
        "parentMessageId": request.parent_message_id,
    }


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
    left_tokens = token_set(" ".join([left.title, left.description]))
    right_tokens = token_set(" ".join([right.title, right.description]))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    same_due = (left.due_text or "").lower() == (right.due_text or "").lower()
    people_overlap = bool(set(left.mentioned_people) & set(right.mentioned_people))
    contained = left.description in right.description or right.description in left.description
    return contained or overlap >= 0.70 or (overlap >= 0.55 and (same_due or people_overlap))


def _merge_task_pair(
    left: ExtractedTaskCandidate,
    right: ExtractedTaskCandidate,
) -> ExtractedTaskCandidate:
    description = (
        left.description if len(left.description) >= len(right.description) else right.description
    )
    title = left.title if len(left.title) <= len(right.title) else right.title
    evidence = list(
        dict.fromkeys(
            [*left.evidence, *right.evidence, "merged_overlapping_task_candidates"]
        )
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
