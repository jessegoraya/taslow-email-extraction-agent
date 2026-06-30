from __future__ import annotations

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.models import EmailExtractionRequest, ExtractedTaskAssignment
from taslow_email_extraction_agent.text_utils import token_set


@step(name="ResultValidationExecutor")
async def validate_assignments(
    assignments: list[ExtractedTaskAssignment],
    settings: Settings,
    request: EmailExtractionRequest | None = None,
) -> list[ExtractedTaskAssignment]:
    deduped_by_source: dict[tuple[str, str, str], ExtractedTaskAssignment] = {}
    for assignment in assignments:
        if assignment.assignee_confidence < settings.assignee_confidence_threshold:
            continue
        if assignment.needs_review:
            continue
        if request and _is_external_sender_assignee(request, assignment):
            continue
        key = (
            assignment.source_task_id,
            assignment.assignee_email.lower(),
            assignment.project_id,
        )
        existing = deduped_by_source.get(key)
        if existing is None or assignment.overall_confidence > existing.overall_confidence:
            deduped_by_source[key] = assignment
    return _dedupe_duplicate_business_tasks(list(deduped_by_source.values()))


def _is_external_sender_assignee(
    request: EmailExtractionRequest,
    assignment: ExtractedTaskAssignment,
) -> bool:
    if not request.from_participant or not request.from_participant.email:
        return False
    sender_email = request.from_participant.email.lower()
    if assignment.assignee_email.lower() != sender_email:
        return False
    sender_domain = _email_domain(sender_email)
    if not sender_domain:
        return False
    return sender_domain not in _tenant_side_domains(request)


def _tenant_side_domains(request: EmailExtractionRequest) -> set[str]:
    domains = {_email_domain(request.mailbox)}
    domains.update(_email_domain(participant.email) for participant in request.all_recipients)
    return {domain for domain in domains if domain}


def _email_domain(email: str) -> str:
    return email.rsplit("@", 1)[1].lower() if "@" in email else ""


def _dedupe_duplicate_business_tasks(
    assignments: list[ExtractedTaskAssignment],
) -> list[ExtractedTaskAssignment]:
    deduped: list[ExtractedTaskAssignment] = []
    for assignment in sorted(assignments, key=_assignment_quality, reverse=True):
        existing_index = next(
            (
                index
                for index, existing in enumerate(deduped)
                if _same_business_task(existing, assignment)
            ),
            None,
        )
        if existing_index is None:
            deduped.append(assignment)
            continue
        deduped[existing_index] = _merge_duplicate_assignment(deduped[existing_index], assignment)
    return sorted(deduped, key=lambda item: item.source_task_id)


def _same_business_task(
    left: ExtractedTaskAssignment,
    right: ExtractedTaskAssignment,
) -> bool:
    if left.project_id != right.project_id:
        return False
    if left.assignee_email.lower() != right.assignee_email.lower():
        return False
    if _has_distinct_due_dates(left, right):
        return False

    left_tokens = _task_intent_tokens(left)
    right_tokens = _task_intent_tokens(right)
    if not left_tokens or not right_tokens:
        return False

    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    left_text = _normalized_task_text(left)
    right_text = _normalized_task_text(right)
    contained = left_text in right_text or right_text in left_text
    title_overlap = _token_overlap(token_set(left.title), token_set(right.title))
    return contained or overlap >= 0.72 or (overlap >= 0.58 and title_overlap >= 0.80)


def _has_distinct_due_dates(
    left: ExtractedTaskAssignment,
    right: ExtractedTaskAssignment,
) -> bool:
    if not left.due_date or not right.due_date:
        return False
    return left.due_date.date() != right.due_date.date()


def _merge_duplicate_assignment(
    left: ExtractedTaskAssignment,
    right: ExtractedTaskAssignment,
) -> ExtractedTaskAssignment:
    winner = left if _assignment_quality(left) >= _assignment_quality(right) else right
    other = right if winner is left else left
    evidence = list(dict.fromkeys([*winner.evidence, *other.evidence, "deduped_duplicate_task"]))
    return winner.model_copy(
        update={
            "description": _longer_text(winner.description, other.description),
            "due_date": winner.due_date or other.due_date,
            "due_date_confidence": winner.due_date_confidence or other.due_date_confidence,
            "evidence": evidence,
        }
    )


def _assignment_quality(assignment: ExtractedTaskAssignment) -> tuple[float, float, float, int]:
    return (
        assignment.overall_confidence,
        assignment.assignee_confidence,
        assignment.scope_confidence or 0.0,
        1 if assignment.due_date else 0,
    )


def _task_intent_tokens(assignment: ExtractedTaskAssignment) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "by",
        "for",
        "from",
        "in",
        "into",
        "me",
        "of",
        "on",
        "or",
        "please",
        "the",
        "to",
        "with",
        "you",
        "your",
    }
    return {
        token
        for token in token_set(_normalized_task_text(assignment))
        if len(token) > 2 and token not in stop_words
    }


def _normalized_task_text(assignment: ExtractedTaskAssignment) -> str:
    return " ".join([assignment.title, assignment.description]).lower().strip()


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _longer_text(left: str, right: str) -> str:
    return left if len(left) >= len(right) else right
