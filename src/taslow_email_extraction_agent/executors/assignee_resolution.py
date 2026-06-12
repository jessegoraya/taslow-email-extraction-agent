from __future__ import annotations

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
)
from taslow_email_extraction_agent.text_utils import token_set


@step(name="AssigneeResolutionExecutor")
async def resolve_assignees(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
    project: ProjectContext,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    visible_recipients = [p for p in request.visible_recipients if p.email]
    project_people_by_email = {person.email: person for person in project.people if person.email}

    if len(visible_recipients) == 1:
        recipient = visible_recipients[0]
        person = project_people_by_email.get(
            recipient.email,
            AssociatedPerson(name=recipient.name, email=recipient.email, aliases="", role=""),
        )
        return [(person, 0.88, ["single_visible_recipient"])]

    task_tokens = token_set(" ".join([task.title, task.description, *task.mentioned_people]))
    matches: list[tuple[AssociatedPerson, float, list[str]]] = []
    recipient_emails = {recipient.email for recipient in visible_recipients}

    for person in project.people:
        score = 0.0
        evidence: list[str] = []
        if person.email in recipient_emails:
            score += 0.35
            evidence.append("recipient_overlap")
        if person.tokens & task_tokens:
            score += 0.55
            evidence.append("task_name_or_alias_match")
        if score:
            matches.append((person, round(min(1.0, score), 3), evidence))

    matches.sort(key=lambda item: item[1], reverse=True)
    if matches:
        top_score = matches[0][1]
        return [match for match in matches if match[1] >= max(0.75, top_score - 0.05)]

    return []
