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
) -> ProjectScore | None:
    if not projects:
        return None

    email_text = " ".join([request.combined_text, *[task.description for task in tasks]])
    participant_emails = {
        p.email for p in [*request.visible_recipients, request.from_participant] if p and p.email
    }
    participant_names = {p.name.lower() for p in request.visible_recipients if p.name}

    scored = [
        ProjectScore(
            project=project,
            result=_score_project(
                project, email_text, participant_emails, participant_names, thread_context
            ),
        )
        for project in projects
    ]
    return max(scored, key=lambda item: item.result.confidence)


def _score_project(
    project: ProjectContext,
    email_text: str,
    participant_emails: set[str],
    participant_names: set[str],
    thread_context: ThreadContext | None,
) -> ProjectMatchResult:
    evidence: list[str] = []

    project_people_emails = {person.email for person in project.people if person.email}
    email_overlap = len(participant_emails & project_people_emails)
    participant_score = min(1.0, email_overlap / max(1, len(participant_emails)))
    if participant_score:
        evidence.append("recipient_or_sender_overlap")

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
    if semantic_score:
        evidence.append("azure_ai_search_project_similarity")

    thread_score = 0.0
    if thread_context and thread_context.project_id == project.project_id:
        thread_score = min(1.0, thread_context.confidence)
        evidence.append("thread_project_history")

    if semantic_score:
        confidence = min(
            1.0,
            (participant_score * 0.32)
            + (people_context_score * 0.15)
            + (lexical_score * 0.18)
            + (semantic_score * 0.20)
            + (thread_score * 0.15),
        )
    else:
        confidence = min(
            1.0,
            (participant_score * 0.42)
            + (people_context_score * 0.18)
            + (lexical_score * 0.25)
            + (thread_score * 0.15),
        )

    return ProjectMatchResult(
        projectId=project.project_id,
        projectName=project.project_name,
        confidence=round(confidence, 3),
        evidence=evidence,
    )
