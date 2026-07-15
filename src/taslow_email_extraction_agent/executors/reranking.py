from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectScope,
)
from taslow_email_extraction_agent.text_utils import (
    has_forwarded_actionable_handoff,
    split_newest_and_quoted_text,
    task_context_text,
)

AssigneeChoice = tuple[AssociatedPerson, float, list[str]]
ScopeChoice = tuple[ProjectScope | None, float, list[str]]
STRONG_DETERMINISTIC_ASSIGNEE_EVIDENCE = {
    "subject_matter_owner_signal",
    "requested_actor_assignment_language",
    "named_request_actor_signal",
    "beneficiary_or_owner_signal",
}
HIGH_DETERMINISTIC_SCOPE_CONFIDENCE = 0.86
LOW_SCOPE_CONFIDENCE_FOR_RERANK = 0.78
STRONG_SCOPE_SEARCH_SCORE = 0.70
STRONG_SCOPE_SEARCH_MARGIN = 0.06
CLOSE_SCOPE_SEARCH_MARGIN = 0.04
GENERIC_DELIVERABLE_WORDS = {
    "binder",
    "deck",
    "deliverable",
    "package",
    "report",
    "reports",
    "summary",
    "submittal",
    "submittals",
    "writeup",
}
SCOPE_EVIDENCE_STOP_WORDS = GENERIC_DELIVERABLE_WORDS | {
    "about",
    "and",
    "area",
    "coordination",
    "for",
    "management",
    "of",
    "project",
    "review",
    "scope",
    "services",
    "support",
    "task",
    "the",
    "work",
}


class AssigneeReranker(Protocol):
    async def rerank_assignees(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        assignees: list[AssigneeChoice],
    ) -> list[AssigneeChoice]:
        """Choose the best assignee from project-associated people only."""


class ScopeReranker(Protocol):
    async def rerank_scope(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        scope: ProjectScope | None,
        confidence: float,
        evidence: list[str],
    ) -> ScopeChoice:
        """Choose the best scope from the selected project's scopes only."""


class NoOpAssigneeReranker:
    async def rerank_assignees(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        assignees: list[AssigneeChoice],
    ) -> list[AssigneeChoice]:
        return assignees


class NoOpScopeReranker:
    async def rerank_scope(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        scope: ProjectScope | None,
        confidence: float,
        evidence: list[str],
    ) -> ScopeChoice:
        return scope, confidence, evidence


class FoundryAssigneeReranker:
    """Model-backed assignee selector constrained to people on the selected project."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def rerank_assignees(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        assignees: list[AssigneeChoice],
    ) -> list[AssigneeChoice]:
        if not self._is_configured or not project.people:
            return assignees
        if _has_strong_deterministic_assignee(assignees):
            return assignees

        try:
            result = await self._complete_json(
                "taslow_assignee_rerank",
                _ASSIGNEE_RERANK_SCHEMA,
                _ASSIGNEE_SYSTEM_PROMPT,
                _assignee_payload(request, task, project, assignees),
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return assignees

        selected_email = str(result.get("assigneeEmail") or "").strip().lower()
        if not selected_email:
            return assignees
        person = next((p for p in project.people if p.email == selected_email), None)
        if not person:
            return assignees

        confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
        if confidence < 0.70:
            return assignees
        existing = next((item for item in assignees if item[0].email == selected_email), None)
        evidence = list(
            dict.fromkeys(
                [
                    *(existing[2] if existing else []),
                    "llm_assignee_reranker",
                    _safe_reason(result.get("rationale")),
                ]
            )
        )
        selected_confidence = max(confidence, existing[1] if existing else confidence)
        return [(person, round(selected_confidence, 3), evidence)]

    @property
    def _is_configured(self) -> bool:
        return bool(
            self._settings.agent_assignee_reranker_enabled
            and self._settings.azure_openai_endpoint
            and self._settings.azure_openai_api_key
            and self._settings.azure_ai_model_deployment_name
        )

    async def _complete_json(
        self,
        schema_name: str,
        schema: dict,
        system_prompt: str,
        payload: dict,
    ) -> dict:
        endpoint = self._settings.azure_openai_endpoint
        deployment = self._settings.azure_ai_model_deployment_name
        assert endpoint is not None
        assert deployment is not None
        url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self._settings.azure_openai_chat_api_version}"
        )
        request_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    "api-key": self._settings.azure_openai_api_key or "",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
            response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])


def _has_strong_deterministic_assignee(assignees: list[AssigneeChoice]) -> bool:
    if not assignees:
        return False
    top_person, top_confidence, top_evidence = assignees[0]
    if not top_person.email or top_confidence < 0.89:
        return False
    return bool(STRONG_DETERMINISTIC_ASSIGNEE_EVIDENCE & set(top_evidence))


def _should_skip_scope_reranker(scope: ProjectScope | None, confidence: float) -> bool:
    if scope is None:
        return False
    if confidence >= HIGH_DETERMINISTIC_SCOPE_CONFIDENCE:
        return True
    if _has_strong_rank_one_scope_margin(scope):
        return True
    return confidence >= LOW_SCOPE_CONFIDENCE_FOR_RERANK and not _has_close_scope_margin(scope)


def _has_strong_rank_one_scope_margin(scope: ProjectScope) -> bool:
    return bool(
        scope.search_rank == 1
        and (scope.search_score or 0.0) >= STRONG_SCOPE_SEARCH_SCORE
        and (scope.search_margin or 0.0) >= STRONG_SCOPE_SEARCH_MARGIN
    )


def _has_close_scope_margin(scope: ProjectScope) -> bool:
    return scope.search_margin is not None and scope.search_margin <= CLOSE_SCOPE_SEARCH_MARGIN


def _generic_deliverable_override_blocked(
    task: ExtractedTaskCandidate,
    result: dict,
    current_scope: ProjectScope | None,
    selected_scope: ProjectScope,
    current_confidence: float,
    rerank_confidence: float,
) -> bool:
    if current_scope is None or current_scope.scope_id == selected_scope.scope_id:
        return False

    text = " ".join(
        [
            task.title,
            task.description,
            str(result.get("rationale") or ""),
        ]
    ).lower()
    if not _has_generic_deliverable_word(text):
        return False

    strong_confidence = rerank_confidence >= 0.90 and rerank_confidence >= current_confidence + 0.08
    return not (strong_confidence and _has_direct_scope_evidence(selected_scope, text))


def _has_generic_deliverable_word(text: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in GENERIC_DELIVERABLE_WORDS)


def _has_direct_scope_evidence(scope: ProjectScope, text: str) -> bool:
    scope_text = f"{scope.title} {scope.description}".lower()
    title = scope.title.strip().lower()
    if len(title) >= 10 and title in text:
        return True
    scope_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9&-]{3,}", scope_text)
        if token not in SCOPE_EVIDENCE_STOP_WORDS
    }
    text_tokens = set(re.findall(r"[a-z0-9][a-z0-9&-]{3,}", text))
    return bool(scope_tokens & text_tokens)


class FoundryScopeReranker(FoundryAssigneeReranker):
    """Model-backed scope selector constrained to scopes on the selected project."""

    async def rerank_scope(
        self,
        request: EmailExtractionRequest,
        task: ExtractedTaskCandidate,
        project: ProjectContext,
        scope: ProjectScope | None,
        confidence: float,
        evidence: list[str],
    ) -> ScopeChoice:
        if not self._scope_configured or not project.scopes:
            return scope, confidence, evidence
        if _should_skip_scope_reranker(scope, confidence):
            return scope, confidence, evidence

        try:
            result = await self._complete_json(
                "taslow_scope_rerank",
                _SCOPE_RERANK_SCHEMA,
                _SCOPE_SYSTEM_PROMPT,
                _scope_payload(request, task, project, scope, confidence, evidence),
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return scope, confidence, evidence

        selected_scope_id = str(result.get("scopeId") or "").strip()
        if not selected_scope_id:
            return scope, confidence, evidence
        selected = next(
            (item for item in project.scopes if item.scope_id == selected_scope_id),
            None,
        )
        if not selected:
            return scope, confidence, evidence
        rerank_confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
        if rerank_confidence < 0.70:
            return scope, confidence, evidence
        if _generic_deliverable_override_blocked(
            task,
            result,
            scope,
            selected,
            confidence,
            rerank_confidence,
        ):
            return (
                scope,
                confidence,
                list(
                    dict.fromkeys(
                        [*evidence, "scope_reranker_generic_deliverable_override_blocked"]
                    )
                ),
            )
        merged_evidence = list(
            dict.fromkeys([*evidence, "llm_scope_reranker", _safe_reason(result.get("rationale"))])
        )
        return selected, round(max(confidence, rerank_confidence), 3), merged_evidence

    @property
    def _scope_configured(self) -> bool:
        return bool(
            self._settings.agent_scope_reranker_enabled
            and self._settings.azure_openai_endpoint
            and self._settings.azure_openai_api_key
            and self._settings.azure_ai_model_deployment_name
        )


def _assignee_payload(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
    project: ProjectContext,
    assignees: list[AssigneeChoice],
) -> dict:
    return {
        "email": _email_payload(request),
        "task": task.model_dump(by_alias=True),
        "project": {
            "projectId": project.project_id,
            "projectName": project.project_name,
            "people": [_person_payload(person) for person in project.people],
        },
        "assignmentPolicy": {
            "allowedAssignees": "Only people in project.people may be selected.",
            "accountableOwnerPreference": (
                "Prefer the person accountable for the implied work, not automatically "
                "the email sender, salutation recipient, or person closest to the task text."
            ),
            "genericRequestFallback": (
                "For generic volunteer/request wording such as 'can someone', "
                "'does anyone know', 'could anyone tell me', 'would someone', "
                "or 'would anyone be willing to', first confirm the phrase is tied "
                "to an actionable work outcome. Same-block Outlook @mentions of "
                "project-associated people outrank weaker recipient heuristics; "
                "otherwise use the single addressed project-associated To recipient, "
                "or the single project manager in To when multiple recipients are present."
            ),
            "unknownPersonRule": (
                "Return null if the best human appears in the email but is not associated "
                "with the selected project."
            ),
        },
        "candidateRoleSignals": [
            _assignee_role_signal(request, task, person) for person in project.people
        ],
        "currentCandidates": [
            {
                "email": person.email,
                "name": person.name,
                "confidence": confidence,
                "evidence": evidence,
            }
            for person, confidence, evidence in assignees
        ],
    }


def _scope_payload(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
    project: ProjectContext,
    scope: ProjectScope | None,
    confidence: float,
    evidence: list[str],
) -> dict:
    newest_body, quoted_context = split_newest_and_quoted_text(request.body_text)
    handoff = has_forwarded_actionable_handoff(request.body_text)
    return {
        "email": _email_payload(request),
        "task": {
            **task.model_dump(by_alias=True),
            "contextText": task_context_text(request.body_text, task.description),
        },
        "forwardedHandoff": {
            "detected": handoff,
            "newestAuthoredText": newest_body,
            "forwardedContextText": quoted_context if handoff else "",
            "policy": (
                "When detected, assignment comes from the newest authored handoff, while "
                "the forwarded context may supply task, Project, and Scope details."
            ),
        },
        "scopeSelectionPolicy": {
            "primaryRule": (
                "Choose the scope for the concrete work outcome requested in the newest "
                "authored message, not the incidental context where the issue was found."
            ),
            "contextVsOutcomeExamples": [
                {
                    "emailContext": "Issue appears in reconciliation output",
                    "workOutcome": "Find and fix mismatched source data",
                    "preferScopeAbout": "data quality or source-data correction",
                    "avoidIfOnlyContext": "reconciliation process",
                },
                {
                    "emailContext": "Issue surfaced during infection control walkthrough",
                    "workOutcome": "Track down and log missing closeout submittal",
                    "preferScopeAbout": "submittals, quality assurance, or closeout",
                    "avoidIfOnlyContext": "infection control",
                },
                {
                    "emailContext": "Template/link was fixed in a parent thread",
                    "workOutcome": "Notify distribution list to refresh local copies",
                    "preferScopeAbout": (
                        "parent/thread scope when available, otherwise communications/governance"
                    ),
                    "avoidIfOnlyContext": "generic reporting deck wording",
                },
                {
                    "emailContext": "Hotel rate variance appears during a CBA cycle review",
                    "workOutcome": "Audit billing, rate sheet, invoice, or reconciliation issue",
                    "preferScopeAbout": "billing, reconciliation, rate audit, or invoice review",
                    "avoidIfOnlyContext": "lodging coordination or travel booking",
                },
                {
                    "emailContext": "Current reply says same scope as below",
                    "workOutcome": "Continue the latest parent-thread work item",
                    "preferScopeAbout": "thread-history scope when supported",
                    "avoidIfOnlyContext": "older quoted task that conflicts with newest request",
                },
                {
                    "emailContext": "Travel reservation system must recognize fare class codes",
                    "workOutcome": "Update reservation, ticketing, or travel configuration",
                    "preferScopeAbout": "official travel reservation and ticketing configuration",
                    "avoidIfOnlyContext": "general travel policy or eTravel reporting",
                },
                {
                    "emailContext": "Emergency generator or MEP equipment requires a load test",
                    "workOutcome": "Coordinate or schedule technical system testing",
                    "preferScopeAbout": "MEP, fire protection, generator, or technical testing",
                    "avoidIfOnlyContext": "general phasing or utility shutdown coordination",
                },
                {
                    "emailContext": "Committee charter language is outdated",
                    "workOutcome": (
                        "Refresh governance, charter, task order, or documentation language"
                    ),
                    "preferScopeAbout": (
                        "governance, charter, documentation, or task-order management"
                    ),
                    "avoidIfOnlyContext": (
                        "dashboard trends or cross-system reconciliation discussion"
                    ),
                },
                {
                    "emailContext": "Dashboard access request is stuck in queue",
                    "workOutcome": "Resolve dashboard, access, reporting, or BI queue issue",
                    "preferScopeAbout": (
                        "business intelligence, dashboard access, reporting, or user access"
                    ),
                    "avoidIfOnlyContext": (
                        "source data quality unless the requested work is data correction"
                    ),
                },
                {
                    "emailContext": "PI planning prep attendance or program board work",
                    "workOutcome": (
                        "Prepare or improve Agile/SAFe PI planning artifacts or participation"
                    ),
                    "preferScopeAbout": "Agile and SAFe product delivery or PI planning",
                    "avoidIfOnlyContext": (
                        "general portfolio governance unless PI planning is incidental"
                    ),
                },
                {
                    "emailContext": "Per-diem rate tables need updating before a review cycle",
                    "workOutcome": "Update per-diem, eTravel, or policy compliance rate tables",
                    "preferScopeAbout": (
                        "eTravel, policy compliance, per-diem, or travel-rate table updates"
                    ),
                    "avoidIfOnlyContext": (
                        "CBA billing reconciliation unless invoice, billing, or reconcile "
                        "language appears"
                    ),
                },
            ],
            "quotedTextRule": (
                "Ignore forwarded or quoted older tasks unless the newest authored text "
                "explicitly asks the recipient to handle, take care of, follow up on, or "
                "otherwise act on the forwarded request."
            ),
        },
        "workOutcomeHints": _work_outcome_hints(request, task),
        "project": {
            "projectId": project.project_id,
            "projectName": project.project_name,
            "scopes": [
                {
                    "scopeId": item.scope_id,
                    "title": item.title,
                    "description": item.description,
                    "searchScore": item.search_score,
                    "searchRank": item.search_rank,
                    "searchMargin": item.search_margin,
                }
                for item in project.scopes
            ],
        },
        "currentScope": {
            "scopeId": scope.scope_id if scope else None,
            "confidence": confidence,
            "evidence": evidence,
            "searchScore": scope.search_score if scope else None,
            "searchRank": scope.search_rank if scope else None,
            "searchMargin": scope.search_margin if scope else None,
        },
    }


def _email_payload(request: EmailExtractionRequest) -> dict:
    newest_body, quoted_context = split_newest_and_quoted_text(request.body_text)
    handoff = has_forwarded_actionable_handoff(request.body_text)
    return {
        "subject": request.subject,
        "bodyText": newest_body,
        "forwardedContextText": quoted_context if handoff else "",
        "forwardedActionableHandoff": handoff,
        "bodyTextRawIncluded": False,
        "from": request.from_participant.model_dump() if request.from_participant else None,
        "to": [recipient.model_dump() for recipient in request.to],
        "cc": [recipient.model_dump() for recipient in request.cc],
        "sentDateTime": request.sent_date_time.isoformat() if request.sent_date_time else None,
    }


def _person_payload(person: AssociatedPerson) -> dict:
    return {
        "email": person.email,
        "name": person.name,
        "aliases": person.aliases,
        "role": person.role,
    }


def _assignee_role_signal(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
    person: AssociatedPerson,
) -> dict:
    text = " ".join(
        [
            request.subject,
            request.body_text,
            task.title,
            task.description,
            *task.mentioned_people,
        ]
    ).lower()
    references = _person_references(person)
    signals: list[str] = []
    if person.email in {recipient.email for recipient in request.visible_recipients}:
        signals.append("visible_recipient")
    if request.from_participant and person.email == request.from_participant.email:
        signals.append("sender")
    if any(reference in text for reference in references):
        signals.append("named_or_alias_mentioned")
    if any(
        re.search(
            rf"\b{re.escape(reference)}(?:'s)?\b.{0, 80}"
            r"\b(?:flagged|mentioned|noticed|thinks|found|identified|is working|wants)\b",
            text,
        )
        for reference in references
    ):
        signals.append("subject_matter_owner")
    if any(
        re.search(
            rf"\b(?:help|make sure|confirm with|brief|walk)\s+{re.escape(reference)}\b",
            text,
        )
        for reference in references
    ):
        signals.append("beneficiary_or_coordination_target")
    return {
        "email": person.email,
        "name": person.name,
        "signals": sorted(set(signals)),
    }


def _person_references(person: AssociatedPerson) -> set[str]:
    references: set[str] = set()
    if person.name:
        references.add(person.name.lower())
        first = person.name.split()[0].lower()
        if len(first) > 2:
            references.add(first)
    if person.aliases:
        for alias in person.aliases.replace(";", ",").split(","):
            cleaned = alias.strip().lower()
            if cleaned:
                references.add(cleaned)
    if person.email:
        handle = person.email.split("@", 1)[0].lower()
        references.add(handle)
        for part in re.split(r"[._-]+", handle):
            if len(part) > 2:
                references.add(part)
    return {reference for reference in references if len(reference) > 2}


def _work_outcome_hints(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
) -> dict:
    newest_body, _quoted_context = split_newest_and_quoted_text(request.body_text)
    text = " ".join(
        [
            request.subject,
            newest_body,
            task_context_text(request.body_text, task.description),
            task.title,
        ]
    ).lower()
    hints: list[str] = []
    if re.search(r"\b(source data|data quality|mismatch|duplicate|record count|schema)\b", text):
        hints.append("data_quality_or_source_data")
    if re.search(r"\b(reconciliation|reconcile|rate sheet|cba)\b", text):
        hints.append("reconciliation_context_or_work")
    if re.search(r"\b(hotel rate|rate audit|rate variance|billing|invoice|cba cycle)\b", text):
        hints.append("billing_reconciliation_or_rate_audit")
    if re.search(r"\b(same scope as below|same project as below|same as below)\b", text):
        hints.append("thread_inheritance_requested")
    if re.search(r"\b(submittal|closeout|binder|as-built|drawings?)\b", text):
        hints.append("submittals_closeout_or_document_control")
    if re.search(r"\b(sign[- ]off|approval|approved|authorize|pending approval)\b", text):
        hints.append("approval_or_signoff")
    if re.search(r"\b(calibration|gauge|maintenance|preventive inspection)\b", text):
        hints.append("calibration_or_maintenance")
    if re.search(r"\b(fare class|reservation|ticketing configuration|premium economy)\b", text):
        hints.append("travel_reservation_ticketing_configuration")
    if re.search(r"\b(generator|load test|mep|fire protection)\b", text):
        hints.append("mep_or_generator_testing")
    if re.search(r"\b(charter|committee charter|task order documentation)\b", text):
        hints.append("governance_charter_or_documentation")
    if re.search(r"\b(dashboard|access request|access queue|reporting access|bi)\b", text):
        hints.append("dashboard_access_or_business_intelligence")
    if re.search(r"\b(pi planning|program board|safe|agile)\b", text):
        hints.append("agile_safe_pi_planning")
    if re.search(r"\b(per[- ]diem|travel rate tables?|rate tables?)\b", text):
        hints.append("per_diem_policy_or_travel_rate_tables")
    if re.search(r"\b(notify|distribution list|refresh|local copies|send a quick note)\b", text):
        hints.append("notification_or_stakeholder_communication")
    if re.search(r"\b(training|walk .* through|new hires?|runbook)\b", text):
        hints.append("training_or_knowledge_transfer")
    if has_forwarded_actionable_handoff(request.body_text):
        hints.append("forwarded_actionable_handoff_context")
    elif re.search(
        r"\bforwarded message|---------- forwarded|old thread|current item is different\b",
        text,
    ):
        hints.append("forwarded_or_quoted_newest_message_only")
    return {
        "hints": sorted(set(hints)),
        "taskTitle": task.title,
        "taskDescription": task.description,
    }


def _safe_reason(value: object) -> str:
    reason = str(value or "").strip().lower().replace(" ", "_")
    return reason[:80] if reason else "llm_reranker_reason_missing"


_ASSIGNEE_SYSTEM_PROMPT = """Select the best assignee for one extracted Taslow task.
You may only choose a person whose email appears in the project people list. Do not choose
external senders, external recipients, or people who are not associated with the project.
Select the accountable owner of the implied work. Distinguish requested actors, accountable
recipients, subject-matter owners, beneficiaries, and external requesters. For generic
volunteer/request wording such as "can someone", "does anyone know", "could anyone tell me",
"would someone", or "would anyone be willing to", first confirm the phrase is tied to an
actionable work outcome. Prefer a qualifying same-block Outlook @mention, otherwise the single
addressed project-associated To recipient, otherwise the single project manager in To when the
message has multiple recipients. Return null when the assignee is not supported by project people
and email/task text."""

_SCOPE_SYSTEM_PROMPT = """Select the best scope for one extracted Taslow task.
You may only choose a scopeId from the selected project's scope list. Prefer the scope whose title
and description best match the concrete work outcome requested in the newest authored message, not
generic project language or incidental context. If the email says an issue was found in one area but
the requested work is to fix another area, choose the scope for the requested work. For forwarded or
quoted content, ignore stale quoted tasks when the newest authored text identifies a different
current item. Return null when no scope is supported."""

_ASSIGNEE_RERANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "assigneeEmail": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["assigneeEmail", "confidence", "rationale"],
}

_SCOPE_RERANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scopeId": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["scopeId", "confidence", "rationale"],
}
