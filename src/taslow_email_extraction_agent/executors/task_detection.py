from __future__ import annotations

import re
from typing import Protocol

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskCandidate

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
    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        """Extract task candidates from the email."""


class HeuristicTaskExtractor:
    """Safe deterministic extractor used until the Foundry-backed extractor is configured."""

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
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
    """Placeholder for a model-backed extractor.

    This adapter preserves the workflow contract. The next implementation slice should wire this
    to Azure AI Foundry using DefaultAzureCredential and force JSON schema output matching
    `ExtractedTaskCandidate`.
    """

    def __init__(self, fallback: TaskExtractor | None = None) -> None:
        self._fallback = fallback or HeuristicTaskExtractor()

    async def extract_tasks(self, request: EmailExtractionRequest) -> list[ExtractedTaskCandidate]:
        return await self._fallback.extract_tasks(request)


@step(name="TaskDetectionExecutor")
async def detect_tasks(
    request: EmailExtractionRequest,
    extractor: TaskExtractor,
) -> list[ExtractedTaskCandidate]:
    return await extractor.extract_tasks(request)
