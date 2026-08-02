from __future__ import annotations

import re
from dataclasses import dataclass, field

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import (
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectMatchResult,
    ThreadContext,
)
from taslow_email_extraction_agent.text_utils import (
    lexical_similarity,
    newest_authored_text,
    token_set,
)


@dataclass(slots=True)
class ProjectScore:
    project: ProjectContext
    result: ProjectMatchResult
    candidate_results: list[ProjectMatchResult] = field(default_factory=list)


@dataclass(slots=True)
class ParticipantOverlapProfile:
    sender_email: str
    sender_shares_recipient_domain: bool
    recipient_emails: set[str]
    matched_emails_by_project: dict[str, set[str]]
    sender_recipient_project_ids: set[str]
    strongest_overlap_project_ids: set[str]


@step(name="ProjectCandidateRetrievalExecutor")
async def retrieve_project_candidates(projects: list[ProjectContext]) -> list[ProjectContext]:
    return projects


@step(name="ProjectScoringExecutor")
async def score_project_candidates(
    request: EmailExtractionRequest,
    tasks: list[ExtractedTaskCandidate],
    projects: list[ProjectContext],
    thread_context: ThreadContext | None,
    threshold: float = 0.80,
) -> ProjectScore | None:
    if not projects:
        return None

    email_text = " ".join([request.combined_text, *[task.description for task in tasks]])
    subject_text = request.subject or ""
    participant_emails = {
        p.email for p in [*request.visible_recipients, request.from_participant] if p and p.email
    }
    participant_names = {p.name.lower() for p in request.visible_recipients if p.name}
    participant_weights = _participant_project_weights(projects, participant_emails)
    sender_email = request.from_participant.email if request.from_participant else ""
    recipient_emails = {
        participant.email for participant in request.visible_recipients if participant.email
    }
    participant_overlap = _participant_overlap_profile(
        projects,
        sender_email,
        recipient_emails,
    )
    client_domain_counts = _client_domain_project_counts(projects)
    unique_scope_titles = _unique_project_scope_titles(projects)
    scope_reference_text = " ".join(
        [
            subject_text,
            newest_authored_text(request.body_text),
            *[task.description for task in tasks],
        ]
    )

    scored = [
        ProjectScore(
            project=project,
            result=_score_project(
                project,
                subject_text,
                email_text,
                participant_emails,
                participant_weights,
                participant_overlap,
                participant_names,
                sender_email,
                client_domain_counts,
                thread_context,
                threshold,
                unique_scope_titles,
                scope_reference_text,
            ),
        )
        for project in projects
    ]
    confidence_best = max(scored, key=lambda item: item.result.confidence)
    participant_best = max(scored, key=_project_primary_selection_key)
    best = max(scored, key=_project_selection_key)
    if (
        participant_best.project.project_id != confidence_best.project.project_id
        and participant_best.result.confidence == confidence_best.result.confidence
    ):
        best.result = best.result.model_copy(
            update={
                "evidence": list(
                    dict.fromkeys(
                        [
                            *best.result.evidence,
                            "project_selection_participant_tiebreak",
                        ]
                    )
                )
            }
        )
    if (
        best.project.project_id != participant_best.project.project_id
        and _project_primary_selection_key(best)
        == _project_primary_selection_key(participant_best)
    ):
        best.result = best.result.model_copy(
            update={
                "evidence": list(
                    dict.fromkeys(
                        [
                            *best.result.evidence,
                            "project_selection_weighted_evidence_tiebreak",
                        ]
                    )
                )
            }
        )
    candidate_results = [
        item.result
        for item in sorted(scored, key=_project_selection_key, reverse=True)
    ]
    return ProjectScore(
        project=best.project,
        result=best.result,
        candidate_results=candidate_results,
    )


def _project_primary_selection_key(
    item: ProjectScore,
) -> tuple[float, bool, int, float, float]:
    result = item.result
    return (
        result.confidence,
        result.sender_recipient_shared_project,
        result.participant_overlap_count,
        result.participant_score or 0.0,
        result.people_context_score or 0.0,
    )


def _project_selection_key(
    item: ProjectScore,
) -> tuple[float, bool, int, float, float, float]:
    result = item.result
    weighted_evidence = (
        ((result.client_domain_score or 0.0) * 0.18)
        + ((result.lexical_score or 0.0) * 0.16)
        + ((result.search_score_normalized or 0.0) * 0.25)
        + (min(1.0, (result.search_margin or 0.0) * 3) * 0.05)
    )
    return (*_project_primary_selection_key(item), weighted_evidence)


def _score_project(
    project: ProjectContext,
    subject_text: str,
    email_text: str,
    participant_emails: set[str],
    participant_weights: dict[str, float],
    participant_overlap: ParticipantOverlapProfile,
    participant_names: set[str],
    sender_email: str,
    client_domain_counts: dict[str, int],
    thread_context: ThreadContext | None,
    threshold: float,
    unique_scope_titles: set[str],
    scope_reference_text: str,
) -> ProjectMatchResult:
    evidence: list[str] = []

    project_people_emails = {person.email for person in project.people if person.email}
    overlapping_emails = participant_emails & project_people_emails
    recipient_overlapping_emails = participant_overlap.recipient_emails & project_people_emails
    participant_overlap_count = len(overlapping_emails)
    recipient_overlap_count = len(recipient_overlapping_emails)
    participant_overlap_ratio = participant_overlap_count / max(1, len(participant_emails))
    sender_project_member = bool(
        participant_overlap.sender_email
        and participant_overlap.sender_email in project_people_emails
    )
    sender_recipient_shared_project = sender_project_member and recipient_overlap_count > 0
    unique_sender_recipient_project = (
        sender_recipient_shared_project
        and participant_overlap.sender_shares_recipient_domain
        and participant_overlap.sender_recipient_project_ids == {project.project_id}
    )
    unique_strongest_participant_overlap = (
        participant_overlap_count >= 2
        and participant_overlap.strongest_overlap_project_ids == {project.project_id}
    )
    participant_score = min(
        1.0,
        sum(participant_weights.get(email, 1.0) for email in overlapping_emails)
        / max(1, len(participant_emails)),
    )
    if participant_score:
        evidence.append("recipient_or_sender_overlap")
    if sender_project_member:
        evidence.append("sender_project_member")
    if recipient_overlap_count:
        evidence.append("recipient_project_member")
    if participant_overlap_count >= 2:
        evidence.append("multi_participant_project_overlap")
    if sender_recipient_shared_project:
        evidence.append("sender_recipient_shared_project")
    if "participant_overlap" in project.candidate_sources:
        evidence.append("participant_project_candidate")
    if (
        "participant_overlap" in project.candidate_sources
        and "azure_ai_search" not in project.candidate_sources
    ):
        evidence.append("participant_project_candidate_expansion")
    if any(participant_weights.get(email, 1.0) < 1.0 for email in overlapping_emails):
        evidence.append("ubiquitous_participant_deweighted")

    name_hits = 0
    email_tokens = token_set(email_text)
    for person in project.people:
        if person.tokens & email_tokens or person.name.lower() in participant_names:
            name_hits += 1
    people_context_score = min(1.0, name_hits / max(1, len(project.people)))
    if people_context_score:
        evidence.append("associated_people_context")

    subject_alias_match = _project_alias_match(subject_text, project)
    if subject_alias_match:
        evidence.append("subject_project_alias_match")
    body_alias_match = (
        _project_alias_match(email_text, project, high_confidence_only=True)
        and not subject_alias_match
    )
    if body_alias_match:
        evidence.append("body_project_alias_match")

    sender_is_external_to_project = bool(sender_email and sender_email not in project_people_emails)
    external_participant_signal = (
        sender_is_external_to_project and people_context_score >= 0.08
    )
    if external_participant_signal:
        evidence.append("external_sender_allowed_with_project_people_context")

    sender_domain = _email_domain(sender_email)
    project_client_domains = set(project.client_domains)
    client_domain_match = bool(sender_domain and sender_domain in project_client_domains)
    client_domain_unique = client_domain_match and client_domain_counts.get(sender_domain, 0) == 1
    client_domain_score = 0.0
    if client_domain_unique:
        client_domain_score = 1.0
        evidence.append("unique_external_client_domain_match")
    elif client_domain_match:
        client_domain_score = 0.6
        evidence.append("external_client_domain_match")

    lexical_score = lexical_similarity(email_text, project.combined_text)
    if lexical_score:
        evidence.append("body_subject_project_similarity")

    semantic_score = project.search_score or 0.0
    search_margin = project.search_margin or 0.0
    if semantic_score:
        evidence.append("azure_ai_search_project_similarity")
    explicit_project_name_match = _has_explicit_project_name_reference(
        project,
        scope_reference_text,
    )
    if explicit_project_name_match:
        evidence.append("explicit_project_name_reference")
    unique_scope_match = _has_unique_scope_title_reference(
        project,
        scope_reference_text,
        unique_scope_titles,
    )
    if unique_scope_match:
        evidence.append("explicit_unique_scope_title_reference")
    scope_title_match = _has_scope_title_reference(project, scope_reference_text)
    if scope_title_match and not unique_scope_match:
        evidence.append("explicit_scope_title_reference")
    open_item_signal = _has_unresolved_work_signal(email_text)
    if open_item_signal:
        evidence.append("implicit_open_item_project_signal")
    unmatched_workstream_reference = _has_unmatched_workstream_reference(
        scope_reference_text,
        explicit_project_name_match=explicit_project_name_match,
        project_alias_match=subject_alias_match or body_alias_match,
        scope_title_match=scope_title_match,
    )
    if unmatched_workstream_reference:
        evidence.append("unmatched_explicit_workstream_reference")

    thread_score = 0.0
    if thread_context and thread_context.project_id == project.project_id:
        thread_score = min(1.0, thread_context.confidence)
        evidence.append("thread_project_history")

    weighted_confidence = min(
        1.0,
        (participant_score * 0.30)
        + (people_context_score * 0.12)
        + (client_domain_score * 0.18)
        + (lexical_score * 0.16)
        + (semantic_score * 0.25)
        + (min(1.0, search_margin * 3) * 0.05)
        + (thread_score * 0.05),
    )
    confidence = weighted_confidence
    decision_reason = "weighted_evidence"

    if explicit_project_name_match:
        confidence = max(
            confidence,
            min(
                0.98,
                threshold
                + 0.12
                + (participant_score * 0.04)
                + (people_context_score * 0.02),
            ),
        )
        decision_reason = "explicit_project_name_reference"
    elif unique_scope_match:
        confidence = max(
            confidence,
            min(
                0.96,
                threshold
                + 0.08
                + (participant_score * 0.04)
                + (people_context_score * 0.02),
            ),
        )
        decision_reason = "explicit_unique_scope_title_reference"
    elif (
        unique_sender_recipient_project
        and not unmatched_workstream_reference
        and not (subject_alias_match or body_alias_match or scope_title_match)
        and thread_score < threshold
    ):
        confidence = max(
            confidence,
            min(
                0.94,
                threshold
                + 0.07
                + min(0.05, max(0, participant_overlap_count - 2) * 0.02)
                + (people_context_score * 0.02),
            ),
        )
        decision_reason = "unique_sender_recipient_project_overlap"
        evidence.append("unique_sender_recipient_project_overlap")
    elif (
        unique_strongest_participant_overlap
        and not unmatched_workstream_reference
        and not (subject_alias_match or body_alias_match or scope_title_match)
        and thread_score < threshold
        and (
            participant_overlap.sender_shares_recipient_domain
            or recipient_overlap_count >= 2
        )
        and (
            people_context_score > 0
            or lexical_score >= 0.04
            or semantic_score >= 0.55
        )
    ):
        confidence = max(
            confidence,
            min(
                0.91,
                threshold
                + 0.08
                + min(0.04, max(0, participant_overlap_count - 2) * 0.02)
                + (people_context_score * 0.02),
            ),
        )
        decision_reason = "strongest_participant_project_overlap"
        evidence.append("strongest_participant_project_overlap")
    elif subject_alias_match and semantic_score >= 0.65 and project.search_rank == 1:
        confidence = max(
            confidence,
            min(
                0.94,
                threshold
                + 0.06
                + (participant_score * 0.07)
                + (people_context_score * 0.03)
                + (lexical_score * 0.02),
            ),
        )
        decision_reason = "subject_alias_search_and_participant_evidence"
    elif subject_alias_match and semantic_score >= 0.60:
        confidence = max(confidence, min(0.88, threshold + 0.04))
        decision_reason = "subject_alias_and_search_evidence"
    elif body_alias_match and semantic_score >= 0.68 and project.search_rank == 1:
        confidence = max(
            confidence,
            min(
                0.88,
                threshold + 0.035 + (participant_score * 0.04) + (people_context_score * 0.015),
            ),
        )
        decision_reason = "body_alias_search_and_participant_evidence"
    elif body_alias_match and semantic_score >= 0.62:
        confidence = max(confidence, min(0.85, threshold + 0.025))
        decision_reason = "body_alias_and_search_evidence"
    elif semantic_score >= 0.82 and participant_score > 0:
        confidence = max(
            confidence,
            min(
                0.96,
                threshold
                + 0.03
                + (participant_score * 0.06)
                + (people_context_score * 0.03)
                + (lexical_score * 0.03),
            ),
        )
        decision_reason = "strong_search_and_participant_evidence"
    elif semantic_score >= 0.88 and (lexical_score >= 0.12 or people_context_score > 0):
        confidence = max(confidence, min(0.92, threshold + 0.02))
        decision_reason = "strong_search_and_text_or_people_evidence"
    elif (
        scope_title_match
        and participant_score >= 0.5
        and semantic_score >= 0.65
    ):
        confidence = max(
            confidence,
            min(
                0.90,
                threshold
                + 0.02
                + (participant_score * 0.04)
                + (people_context_score * 0.02),
            ),
        )
        decision_reason = "explicit_scope_and_participant_evidence"
    elif (
        participant_score >= 0.5
        and semantic_score >= 0.65
        and (
            subject_alias_match
            or body_alias_match
            or client_domain_match
            or thread_score >= threshold
        )
    ):
        confidence = max(confidence, min(0.90, threshold + 0.01))
        decision_reason = "participant_evidence_with_moderate_search"
    elif (
        open_item_signal
        and project.search_rank == 1
        and semantic_score >= 0.68
        and participant_score >= 0.5
        and search_margin >= 0.02
        and lexical_score >= 0.08
    ):
        confidence = max(
            confidence,
            min(
                0.88,
                threshold + 0.015 + (people_context_score * 0.02) + (lexical_score * 0.02),
            ),
        )
        decision_reason = "open_item_search_and_participant_evidence"
    elif client_domain_unique and project.search_rank == 1 and semantic_score >= 0.52:
        confidence = max(
            confidence,
            min(
                0.90,
                threshold
                + 0.025
                + (participant_score * 0.05)
                + (people_context_score * 0.02)
                + (lexical_score * 0.02),
            ),
        )
        decision_reason = "unique_client_domain_and_search_evidence"
    elif (
        client_domain_match
        and semantic_score >= 0.65
        and (participant_score > 0 or lexical_score >= 0.05 or people_context_score > 0)
    ):
        confidence = max(
            confidence,
            min(0.86, threshold + 0.015 + (client_domain_score * 0.02)),
        )
        decision_reason = "client_domain_with_supporting_project_evidence"
    elif external_participant_signal and semantic_score >= 0.70 and lexical_score >= 0.04:
        confidence = max(
            confidence,
            min(0.82, threshold + 0.005 + (people_context_score * 0.02)),
        )
        decision_reason = "external_sender_project_people_and_search_evidence"
    elif external_participant_signal and lexical_score >= 0.12:
        confidence = max(confidence, min(0.84, threshold + 0.01))
        decision_reason = "external_sender_project_people_and_text_evidence"
    elif thread_score >= threshold and (semantic_score >= 0.55 or participant_score > 0):
        confidence = max(confidence, min(0.90, thread_score))
        decision_reason = "thread_history_supported"
    elif (
        not semantic_score
        and participant_score >= 0.5
        and (people_context_score > 0 or lexical_score >= 0.08)
    ):
        confidence = max(confidence, min(0.88, threshold + 0.01))
        decision_reason = "participant_evidence_without_search"
    elif semantic_score and participant_score == 0 and lexical_score < 0.08:
        decision_reason = "search_without_supporting_evidence"
    elif not semantic_score and participant_score == 0 and lexical_score < 0.08:
        decision_reason = "weak_project_evidence"

    strong_body_alias_match = body_alias_match and (
        participant_score >= 0.5
        or people_context_score >= 0.65
        or (project.search_rank == 1 and semantic_score >= 0.78 and search_margin >= 0.05)
    )
    strong_project_anchor = (
        explicit_project_name_match
        or unique_scope_match
        or (scope_title_match and participant_score >= 0.5)
        or subject_alias_match
        or thread_score >= threshold
        or (
            open_item_signal
            and project.search_rank == 1
            and semantic_score >= 0.68
            and participant_score >= 0.5
            and search_margin >= 0.02
            and lexical_score >= 0.08
        )
        or client_domain_unique
        or (
            client_domain_match
            and (semantic_score >= 0.65 or participant_score > 0 or lexical_score >= 0.08)
        )
        or strong_body_alias_match
    )
    if confidence >= threshold and sender_is_external_to_project and not strong_project_anchor:
        confidence = min(confidence, threshold - 0.01)
        decision_reason = "weak_external_project_anchor"
        evidence.append("weak_external_project_anchor")

    return ProjectMatchResult(
        projectId=project.project_id,
        projectName=project.project_name,
        confidence=round(confidence, 3),
        evidence=evidence,
        searchScoreRaw=project.search_score_raw,
        searchScoreNormalized=round(semantic_score, 4) if semantic_score else None,
        searchRank=project.search_rank,
        searchMargin=project.search_margin,
        participantScore=round(participant_score, 3),
        participantOverlapCount=participant_overlap_count,
        participantOverlapRatio=round(participant_overlap_ratio, 3),
        recipientOverlapCount=recipient_overlap_count,
        senderProjectMember=sender_project_member,
        senderRecipientSharedProject=sender_recipient_shared_project,
        peopleContextScore=round(people_context_score, 3),
        clientDomainScore=round(client_domain_score, 3) if client_domain_score else None,
        lexicalScore=round(lexical_score, 3),
        threshold=threshold,
        decisionReason=decision_reason,
    )


def _participant_project_weights(
    projects: list[ProjectContext],
    participant_emails: set[str],
) -> dict[str, float]:
    """Down-weight participants associated to a disproportionate number of candidates."""
    if not projects or not participant_emails:
        return {}

    project_count = len(projects)
    associations = {email: 0 for email in participant_emails}
    for project in projects:
        project_emails = {person.email for person in project.people if person.email}
        for email in participant_emails & project_emails:
            associations[email] += 1

    weights: dict[str, float] = {}
    for email, count in associations.items():
        ratio = count / project_count if project_count else 0.0
        weights[email] = 0.15 if count >= 3 and ratio >= 0.50 else 1.0
    return weights


def _participant_overlap_profile(
    projects: list[ProjectContext],
    sender_email: str,
    recipient_emails: set[str],
) -> ParticipantOverlapProfile:
    participant_emails = set(recipient_emails)
    if sender_email:
        participant_emails.add(sender_email)
    recipient_domains = {_email_domain(email) for email in recipient_emails if email}
    sender_shares_recipient_domain = bool(
        sender_email and _email_domain(sender_email) in recipient_domains
    )

    matched_emails_by_project: dict[str, set[str]] = {}
    sender_recipient_project_ids: set[str] = set()
    for project in projects:
        project_emails = {person.email for person in project.people if person.email}
        matched_emails = participant_emails & project_emails
        matched_emails_by_project[project.project_id] = matched_emails
        if sender_email in project_emails and recipient_emails & project_emails:
            sender_recipient_project_ids.add(project.project_id)

    max_overlap = max(
        (len(emails) for emails in matched_emails_by_project.values()),
        default=0,
    )
    strongest_overlap_project_ids = {
        project_id
        for project_id, matched_emails in matched_emails_by_project.items()
        if max_overlap >= 2 and len(matched_emails) == max_overlap
    }
    return ParticipantOverlapProfile(
        sender_email=sender_email,
        sender_shares_recipient_domain=sender_shares_recipient_domain,
        recipient_emails=recipient_emails,
        matched_emails_by_project=matched_emails_by_project,
        sender_recipient_project_ids=sender_recipient_project_ids,
        strongest_overlap_project_ids=strongest_overlap_project_ids,
    )


def _client_domain_project_counts(projects: list[ProjectContext]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for project in projects:
        for domain in set(project.client_domains):
            counts[domain] = counts.get(domain, 0) + 1
    return counts


def _unique_project_scope_titles(projects: list[ProjectContext]) -> set[str]:
    owners: dict[str, set[str]] = {}
    for project in projects:
        for scope in project.scopes:
            title = _scope_reference_key(scope.title)
            if (
                len(title) < 12
                or len(title.split()) < 2
                or not _scope_title_is_distinctive(title)
            ):
                continue
            owners.setdefault(title, set()).add(project.project_id)
    return {title for title, project_ids in owners.items() if len(project_ids) == 1}


def _has_explicit_project_name_reference(project: ProjectContext, text: str) -> bool:
    project_name = _scope_reference_key(project.project_name)
    text_key = _scope_reference_key(text)
    if len(project_name) < 8 or len(project_name.split()) < 2 or not text_key:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(project_name)}(?![a-z0-9])",
            text_key,
        )
    )


def _has_unique_scope_title_reference(
    project: ProjectContext,
    text: str,
    unique_scope_titles: set[str],
) -> bool:
    text_key = _scope_reference_key(text)
    if not text_key:
        return False
    return any(
        title in unique_scope_titles
        and re.search(rf"(?<![a-z0-9]){re.escape(title)}(?![a-z0-9])", text_key)
        for title in (_scope_reference_key(scope.title) for scope in project.scopes)
        if title
    )


def _has_scope_title_reference(project: ProjectContext, text: str) -> bool:
    text_key = _scope_reference_key(text)
    if not text_key:
        return False
    return any(
        len(title) >= 8
        and len(title.split()) >= 2
        and re.search(rf"(?<![a-z0-9]){re.escape(title)}(?![a-z0-9])", text_key)
        for title in (_scope_reference_key(scope.title) for scope in project.scopes)
        if title
    )


def _scope_reference_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _scope_title_is_distinctive(title: str) -> bool:
    tokens = [token for token in _scope_reference_key(title).split() if len(token) > 1]
    return any(
        token not in _GENERIC_SCOPE_TITLE_TOKENS
        and (len(token) >= 4 or any(character.isdigit() for character in token))
        for token in tokens
    )


def _email_domain(email: str) -> str:
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower()


def _has_unresolved_work_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:open\s+item|remaining\s+gap|still\s+outstanding|"
            r"at\s+risk\s+of\s+slipping|yours\s+to\s+drive|"
            r"needs?\s+to\s+be\s+(?:resolved|cleared|closed|updated|preserved))\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_unmatched_workstream_reference(
    text: str,
    *,
    explicit_project_name_match: bool,
    project_alias_match: bool,
    scope_title_match: bool,
) -> bool:
    if explicit_project_name_match or project_alias_match or scope_title_match:
        return False
    return bool(
        re.search(
            r"\b(?:this|that|the|another|new)\s+"
            r"(?:project|program|workstream|initiative)\b|"
            r"\b(?:project|program|workstream|initiative)\s+"
            r"(?:named|called|code[- ]named)\b",
            text,
            re.IGNORECASE,
        )
    )


_GENERIC_ACRONYM_SUFFIXES = {
    "administration",
    "development",
    "engineering",
    "maintenance",
    "management",
    "operations",
    "program",
    "services",
    "support",
    "system",
    "systems",
}


_COMMON_ALIAS_TOKENS = {
    "and",
    "building",
    "department",
    "development",
    "emergency",
    "engineering",
    "for",
    "in",
    "inspection",
    "maintenance",
    "management",
    "observation",
    "of",
    "operations",
    "program",
    "project",
    "renovate",
    "reporting",
    "services",
    "support",
    "system",
    "systems",
    "the",
    "unit",
}


_GENERIC_SCOPE_TITLE_TOKENS = {
    "action",
    "actions",
    "administration",
    "analysis",
    "and",
    "contract",
    "contractor",
    "deliverable",
    "deliverables",
    "delivery",
    "facility",
    "format",
    "frequency",
    "general",
    "investigation",
    "investigations",
    "management",
    "medium",
    "operational",
    "operations",
    "personnel",
    "plan",
    "program",
    "project",
    "quality",
    "report",
    "reporting",
    "requirement",
    "requirements",
    "review",
    "section",
    "security",
    "service",
    "services",
    "support",
    "task",
    "tasks",
    "team",
}


def _project_alias_match(
    text: str,
    project: ProjectContext,
    high_confidence_only: bool = False,
) -> bool:
    if not text.strip() or not project.project_name.strip():
        return False
    alias_text = _alias_key(text)
    text_tokens = token_set(text)
    for alias in _project_aliases(
        project.project_name,
        include_generated_prefixes=not high_confidence_only,
        include_upper_tokens=not high_confidence_only,
    ):
        if len(alias) < 3:
            continue
        alias_key = _alias_key(alias)
        if not alias_key:
            continue
        if alias.lower() in text_tokens:
            return True
        if _alias_matches_text(alias, alias_key, text, alias_text):
            return True
    return False


def _project_aliases(
    project_name: str,
    include_generated_prefixes: bool = True,
    include_upper_tokens: bool = True,
) -> set[str]:
    aliases: set[str] = set()
    for match in re.finditer(r"\(([^)]+)\)", project_name):
        for alias in re.split(r"[,;/]", match.group(1)):
            cleaned = alias.strip()
            if 2 <= len(cleaned) <= 20 and re.search(r"[A-Za-z]", cleaned):
                aliases.add(cleaned)

    words = [word for word in re.findall(r"[A-Za-z0-9]+", project_name) if len(word) > 1]
    _add_acronym_aliases(aliases, words, include_prefixes=include_generated_prefixes)

    trimmed_words = words.copy()
    while trimmed_words and trimmed_words[-1].lower() in _GENERIC_ACRONYM_SUFFIXES:
        trimmed_words.pop()
    _add_acronym_aliases(
        aliases,
        trimmed_words,
        include_prefixes=include_generated_prefixes,
    )

    if include_upper_tokens:
        for token in words:
            if (
                3 <= len(token) <= 12
                and token.upper() == token
                and token.lower() not in _COMMON_ALIAS_TOKENS
                and re.search(r"[A-Z]", token)
            ):
                aliases.add(token)
    return {alias for alias in aliases if len(_alias_key(alias)) >= 3}


def _add_acronym_aliases(
    aliases: set[str],
    words: list[str],
    include_prefixes: bool,
) -> None:
    if len(words) < 2:
        return
    acronym = "".join(word[0] for word in words)
    if len(acronym) < 3:
        return
    aliases.add(acronym)
    if include_prefixes:
        for length in range(3, min(6, len(acronym)) + 1):
            aliases.add(acronym[:length])


def _alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _alias_matches_text(alias: str, alias_key: str, text: str, alias_text: str) -> bool:
    compact_pattern = rf"(?<![a-z0-9]){re.escape(alias_key)}(?![a-z0-9])"
    if re.search(compact_pattern, alias_text):
        return True

    alias_parts = [part for part in re.split(r"[^a-z0-9]+", alias.lower()) if part]
    if len(alias_parts) <= 1:
        return False
    flexible_pattern = (
        r"(?<![a-z0-9])"
        + r"[^a-z0-9]+".join(re.escape(part) for part in alias_parts)
        + r"(?![a-z0-9])"
    )
    return bool(re.search(flexible_pattern, text.lower()))
