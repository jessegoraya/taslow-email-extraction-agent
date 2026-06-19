from __future__ import annotations

from dataclasses import dataclass

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import (
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectMatchResult,
    ThreadContext,
)
from taslow_email_extraction_agent.text_utils import lexical_similarity, token_set


@dataclass(slots=True)
class ProjectScore:
    project: ProjectContext
    result: ProjectMatchResult


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
    participant_emails = {
        p.email for p in [*request.visible_recipients, request.from_participant] if p and p.email
    }
    participant_names = {p.name.lower() for p in request.visible_recipients if p.name}
    participant_weights = _participant_project_weights(projects, participant_emails)

    scored = [
        ProjectScore(
            project=project,
            result=_score_project(
                project,
                email_text,
                participant_emails,
                participant_weights,
                participant_names,
                thread_context,
                threshold,
            ),
        )
        for project in projects
    ]
    return max(scored, key=lambda item: item.result.confidence)


def _score_project(
    project: ProjectContext,
    email_text: str,
    participant_emails: set[str],
    participant_weights: dict[str, float],
    participant_names: set[str],
    thread_context: ThreadContext | None,
    threshold: float,
) -> ProjectMatchResult:
    evidence: list[str] = []

    project_people_emails = {person.email for person in project.people if person.email}
    overlapping_emails = participant_emails & project_people_emails
    participant_score = min(
        1.0,
        sum(participant_weights.get(email, 1.0) for email in overlapping_emails)
        / max(1, len(participant_emails)),
    )
    if participant_score:
        evidence.append("recipient_or_sender_overlap")
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

    lexical_score = lexical_similarity(email_text, project.combined_text)
    if lexical_score:
        evidence.append("body_subject_project_similarity")

    semantic_score = project.search_score or 0.0
    search_margin = project.search_margin or 0.0
    if semantic_score:
        evidence.append("azure_ai_search_project_similarity")

    thread_score = 0.0
    if thread_context and thread_context.project_id == project.project_id:
        thread_score = min(1.0, thread_context.confidence)
        evidence.append("thread_project_history")

    weighted_confidence = min(
        1.0,
        (participant_score * 0.30)
        + (people_context_score * 0.12)
        + (lexical_score * 0.16)
        + (semantic_score * 0.32)
        + (min(1.0, search_margin * 3) * 0.05)
        + (thread_score * 0.05),
    )
    confidence = weighted_confidence
    decision_reason = "weighted_evidence"

    if semantic_score >= 0.82 and participant_score > 0:
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
    elif participant_score >= 0.5 and semantic_score >= 0.65:
        confidence = max(confidence, min(0.90, threshold + 0.01))
        decision_reason = "participant_evidence_with_moderate_search"
    elif thread_score >= threshold and (semantic_score >= 0.55 or participant_score > 0):
        confidence = max(confidence, min(0.90, thread_score))
        decision_reason = "thread_history_supported"
    elif not semantic_score and participant_score >= 0.5 and (
        people_context_score > 0 or lexical_score >= 0.08
    ):
        confidence = max(confidence, min(0.88, threshold + 0.01))
        decision_reason = "participant_evidence_without_search"
    elif semantic_score and participant_score == 0 and lexical_score < 0.08:
        decision_reason = "search_without_supporting_evidence"
    elif not semantic_score and participant_score == 0 and lexical_score < 0.08:
        decision_reason = "weak_project_evidence"

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
        peopleContextScore=round(people_context_score, 3),
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
