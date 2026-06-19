from __future__ import annotations

import re

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
    task_text = " ".join([task.title, task.description, *task.mentioned_people])
    task_tokens = token_set(task_text)

    explicit_matches = _explicit_assignment_matches(task_text, project.people)
    if explicit_matches:
        return explicit_matches

    if len(visible_recipients) == 1:
        recipient = visible_recipients[0]
        person = project_people_by_email.get(recipient.email)
        if person:
            return [(person, 0.88, ["single_visible_project_recipient"])]

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


def _explicit_assignment_matches(
    task_text: str,
    project_people: list[AssociatedPerson],
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    matches: list[tuple[AssociatedPerson, float, list[str]]] = []
    normalized = task_text.strip()

    for person in project_people:
        if not person.email:
            continue
        evidence: list[str] = []
        score = 0.0

        for reference in _person_reference_variants(person):
            reference_pattern = re.escape(reference)
            direct_address_pattern = rf"^\s*{reference_pattern}\b\s*,"
            delegated_pattern = rf"\b(?:have|ask|tell)\s+{reference_pattern}\b"
            named_action_pattern = (
                rf"\b{reference_pattern}\b(?:\s|,)*(?:can you|please|need you|"
                r"update|review|prepare|send|coordinate)\b"
            )
            direct_delegation_pattern = (
                r"\b(?:have|ask|tell)\s+"
                rf"(?!{reference_pattern}\b)[a-z][a-z'-]*\b.*\b"
                r"(?:can you|please|need you|update|review|prepare|send|coordinate)\b"
            )

            if re.search(direct_address_pattern, normalized, re.IGNORECASE):
                if not re.search(direct_delegation_pattern, normalized, re.IGNORECASE):
                    score = max(score, 0.96)
                    evidence.append("direct_address_assignment")
            if re.search(delegated_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.98)
                evidence.append("delegated_assignment_language")
            if re.search(named_action_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.88)
                evidence.append("named_person_action_language")

        if score:
            matches.append((person, score, sorted(set(evidence))))

    matches.sort(key=lambda item: item[1], reverse=True)
    if not matches:
        return []
    top_score = matches[0][1]
    return [match for match in matches if match[1] >= max(0.80, top_score - 0.03)]


def _person_reference_variants(person: AssociatedPerson) -> set[str]:
    variants: set[str] = set()
    if person.name:
        variants.add(person.name.lower())
        first_name = person.name.split()[0].lower()
        if len(first_name) > 2:
            variants.add(first_name)
    if person.aliases:
        for alias in person.aliases.replace(";", ",").split(","):
            cleaned = alias.strip().lower()
            if cleaned:
                variants.add(cleaned)
    if person.email:
        handle = person.email.split("@", 1)[0].lower()
        variants.add(handle)
        for part in re.split(r"[._-]+", handle):
            if len(part) > 2:
                variants.add(part)
    return {variant for variant in variants if len(variant) > 2}
