from __future__ import annotations

import re

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
)
from taslow_email_extraction_agent.text_utils import newest_authored_text, token_set

GENERIC_REQUEST_RE = re.compile(
    r"\b(?:"
    r"can\s+someone|"
    r"could\s+someone(?:\s+please)?|"
    r"would\s+someone|"
    r"someone\s+needs\s+to|"
    r"does\s+anyone\s+know|"
    r"could\s+anyone\s+tell\s+me|"
    r"is\s+there\s+anyone\s+who\s+can|"
    r"would\s+anyone\s+be\s+willing\s+to|"
    r"is\s+it\s+possible\s+for\s+someone\s+to|"
    r"may\s+i\s+ask\s+someone\s+to|"
    r"would\s+it\s+be\s+possible\s+for\s+anyone\s+to"
    r")\b",
    re.IGNORECASE,
)

ACTIONABLE_OUTCOME_RE = re.compile(
    r"\b(?:"
    r"perform|produce|send|update|confirm|schedule|review|complete|deliver|document|"
    r"follow\s+up|provide|prepare|draft|create|revise|reconcile|resolve|"
    r"check|validate|verify|investigate|analy[sz]e|clean\s+up|coordinate|"
    r"brief|upload|attach|share|route|submit|approve|close|fix|summari[sz]e|tell\s+me|"
    r"let\s+me\s+know|answer|get\b.{0,40}\bdone"
    r")\b",
    re.IGNORECASE,
)

PURE_KNOWLEDGE_RE = re.compile(
    r"\bdoes\s+anyone\s+know\s+(?:if|whether)\b",
    re.IGNORECASE,
)


@step(name="AssigneeResolutionExecutor")
async def resolve_assignees(
    request: EmailExtractionRequest,
    task: ExtractedTaskCandidate,
    project: ProjectContext,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    project = _with_request_participant_display_names(request, project)
    visible_recipients = [p for p in request.visible_recipients if p.email]
    project_people_by_email = {person.email: person for person in project.people if person.email}
    task_text = " ".join([task.title, task.description, *task.mentioned_people])
    authored_text = newest_authored_text(request.body_text)
    local_assignment_text = _task_local_assignment_text(authored_text, task)
    email_task_text = " ".join(
        [
            request.subject,
            local_assignment_text or authored_text,
            task_text,
        ]
    )
    task_tokens = token_set(task_text)

    named_deliverable_matches = _named_deliverable_from_matches(
        local_assignment_text,
        project.people,
    )
    if not named_deliverable_matches:
        named_deliverable_matches = _named_deliverable_from_matches(
            task_text,
            project.people,
        )
    if named_deliverable_matches:
        return named_deliverable_matches

    explicit_matches = _explicit_assignment_matches(local_assignment_text, project.people)
    if not explicit_matches:
        explicit_matches = _explicit_assignment_matches(task_text, project.people)
    if not explicit_matches and not local_assignment_text:
        explicit_matches = _explicit_assignment_matches(authored_text, project.people)
    if explicit_matches:
        if _explicit_matches_are_soft_direct(explicit_matches):
            owner_matches = _accountable_owner_matches(request, email_task_text, project)
            if owner_matches and _has_strong_owner_evidence(owner_matches):
                return owner_matches
        return explicit_matches

    if _has_generic_request_phrase(email_task_text) and not _has_actionable_outcome(
        email_task_text
    ):
        return []

    accountable_matches = _accountable_owner_matches(request, email_task_text, project)
    if accountable_matches:
        return accountable_matches
    if _has_generic_request_phrase(email_task_text):
        return []

    directional_matches = _directional_you_matches(request, email_task_text, project.people)
    if directional_matches:
        return directional_matches

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
        if _is_context_reviewer_or_delivery_target_only(email_task_text, person):
            continue
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


def _with_request_participant_display_names(
    request: EmailExtractionRequest,
    project: ProjectContext,
) -> ProjectContext:
    display_names = {
        participant.email: participant.name.strip()
        for participant in [
            request.from_participant,
            *request.to,
            *request.cc,
            *request.bcc,
        ]
        if participant and participant.email and participant.name.strip()
    }
    if not display_names:
        return project

    def enrich(person: AssociatedPerson) -> AssociatedPerson:
        display_name = display_names.get(person.email)
        if not display_name or not _is_mailbox_handle_name(person.name, person.email):
            return person
        aliases = [item.strip() for item in person.aliases.split(",") if item.strip()]
        if person.name.strip() and person.name.strip().lower() not in {
            item.lower() for item in aliases
        }:
            aliases.append(person.name.strip())
        return person.model_copy(
            update={
                "name": display_name,
                "aliases": ",".join(aliases),
            }
        )

    return project.model_copy(
        update={
            "associated_people": [enrich(person) for person in project.associated_people],
            "associated_managers": [enrich(person) for person in project.associated_managers],
        }
    )


def _is_mailbox_handle_name(name: str, email: str) -> bool:
    normalized_name = re.sub(r"[^a-z0-9]+", "", (name or "").lower())
    normalized_handle = re.sub(
        r"[^a-z0-9]+",
        "",
        (email or "").split("@", 1)[0].lower(),
    )
    return not normalized_name or normalized_name == normalized_handle


def _explicit_assignment_matches(
    task_text: str,
    project_people: list[AssociatedPerson],
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    matches: list[tuple[AssociatedPerson, float, list[str]]] = []
    normalized = task_text.strip()

    for person in project_people:
        if not person.email:
            continue
        if _is_context_reviewer_or_delivery_target_only(
            normalized,
            person,
            include_coordination_targets=False,
        ):
            continue
        evidence: list[str] = []
        score = 0.0

        for reference in _person_reference_variants(person):
            reference_pattern = re.escape(reference)
            direct_address_pattern = (
                rf"^\s*{reference_pattern}\b\s*,"
                r".{0,80}\b(?:can|could|would)\s+you\b|"
                rf"^\s*{reference_pattern}\b\s*,"
                r".{0,80}\b(?:please|need\s+you\s+to)\b"
            )
            delegated_pattern = rf"\b(?:have|ask|tell)\s+{reference_pattern}\b"
            delegated_request_pattern = (
                rf"\b(?:can|could|would)\s+you\s+(?:have|ask|tell)\s+{reference_pattern}\b"
            )
            need_person_to_pattern = rf"\b(?:we\s+)?need\s+{reference_pattern}\s+to\b"
            please_ask_person_pattern = (
                rf"\bplease\s+(?:have|ask|tell)\s+{reference_pattern}\s+to\b"
            )
            wants_person_to_pattern = rf"\bwants\s+{reference_pattern}\s+to\b"
            named_owner_drive_pattern = (
                rf"(?:@{reference_pattern}|{reference_pattern})\b\s*,?"
                r".{0,80}\b(?:this\s+is\s+yours\s+to\s+(?:drive|own|handle)|"
                r"you\s+(?:own|drive|handle)\s+this|"
                r"please\s+(?:drive|own|handle|take)\s+this|"
                r"(?:can|could|would)\s+you\s+(?:drive|own|handle|take)\s+this)"
            )
            modal_person_action_pattern = (
                rf"\b(?:can|could|should|will)\s+{reference_pattern}\b\s+"
                r"(?:confirm|update|review|prepare|send|coordinate|check|handle|"
                r"reconcile|resolve|brief|walk|find|complete|document|summari[sz]e)\b"
            )
            named_person_modal_action_pattern = (
                rf"\b{reference_pattern}\b\s+(?:should|will|must|is\s+to)\s+"
                r"(?:perform|produce|send|update|confirm|schedule|review|complete|"
                r"deliver|document|follow\s+up|provide|prepare|draft|create|revise|reconcile|"
                r"resolve|check|validate|verify|investigate|analy[sz]e|clean\s+up|"
                r"coordinate|brief|upload|attach|share|route|submit|approve|close|fix|"
                r"summari[sz]e)\b"
            )
            named_action_pattern = (
                rf"\b{reference_pattern}\b(?:\s|,)*(?:can you|please|need you|"
                r"update|review|prepare|send|coordinate|document|summari[sz]e)\b"
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
            if re.search(delegated_request_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.98)
                evidence.append("delegated_assignment_language")
            if re.search(need_person_to_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.97)
                evidence.append("need_named_person_to_act")
            if re.search(please_ask_person_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.97)
                evidence.append("delegated_assignment_language")
            if re.search(wants_person_to_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.96)
                evidence.append("requested_actor_assignment_language")
            if re.search(named_owner_drive_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.97)
                evidence.append("named_owner_drive_assignment")
            if re.search(modal_person_action_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.94)
                evidence.append("modal_named_actor_assignment")
            if re.search(named_person_modal_action_pattern, normalized, re.IGNORECASE):
                score = max(score, 0.95)
                evidence.append("named_person_modal_assignment")
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


def _explicit_matches_are_soft_direct(
    matches: list[tuple[AssociatedPerson, float, list[str]]],
) -> bool:
    soft_evidence = {"direct_address_assignment", "named_person_action_language"}
    return bool(matches) and all(
        set(evidence).issubset(soft_evidence) for _p, _s, evidence in matches
    )


def _task_local_assignment_text(
    authored_text: str,
    task: ExtractedTaskCandidate,
) -> str:
    grounded_evidence = _best_grounded_task_evidence(authored_text, task)
    if grounded_evidence:
        return grounded_evidence

    due_text = (task.due_text or "").strip()
    if not authored_text or not due_text:
        return ""

    anchor_start = authored_text.lower().find(due_text.lower())
    if anchor_start < 0:
        return ""

    boundaries = [0]
    for match in re.finditer(
        r"(?:\r?\n){2,}|(?<=[.!?])\s+(?=[A-ZI])|"
        r",\s+(?:and|also|while|then)\s+(?=[A-Z])",
        authored_text,
        re.IGNORECASE,
    ):
        boundaries.extend([match.start(), match.end()])
    boundaries.append(len(authored_text))
    start = max(boundary for boundary in boundaries if boundary <= anchor_start)
    end = min(boundary for boundary in boundaries if boundary > anchor_start)
    return authored_text[start:end].strip()


def _best_grounded_task_evidence(
    authored_text: str,
    task: ExtractedTaskCandidate,
) -> str:
    if not authored_text or not task.evidence:
        return ""

    authored_lower = authored_text.lower()
    task_tokens = token_set(" ".join([task.title, task.description]))
    due_text = (task.due_text or "").strip().lower()
    candidates: list[tuple[float, int, str]] = []

    for evidence in task.evidence:
        snippet = evidence.strip().strip("\"'“”‘’").strip()
        if len(snippet) < 12:
            continue
        start = authored_lower.find(snippet.lower())
        if start < 0:
            continue

        snippet_tokens = token_set(snippet)
        overlap = (
            len(task_tokens & snippet_tokens) / len(task_tokens)
            if task_tokens
            else 0.0
        )
        due_bonus = 1.0 if due_text and due_text in snippet.lower() else 0.0
        grounded_text = authored_text[start : start + len(snippet)]
        candidates.append((due_bonus + overlap, len(snippet), grounded_text))

    if not candidates:
        return ""
    return max(candidates, key=lambda item: (item[0], item[1]))[2].strip()


def _named_deliverable_from_matches(
    task_local_text: str,
    project_people: list[AssociatedPerson],
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    normalized = task_local_text.strip()
    if not normalized:
        return []

    deliverable = (
        r"(?:analysis|brief|deliverable|draft|follow[- ]up|package|readout|"
        r"report|response|review|summary|update|write[- ]?up)"
    )
    request_prefix = (
        r"(?:i|we)\s+(?:need|want|would\s+like)|"
        r"(?:i|we)['’]d\s+like|"
        r"(?:need|want|request|please\s+(?:provide|prepare|produce|send))"
    )
    action_prefix = r"(?:create|deliver|draft|prepare|produce|provide|send|summari[sz]e)"

    matches: list[tuple[AssociatedPerson, float, list[str]]] = []
    for person in project_people:
        if not person.email:
            continue
        for reference in _person_reference_variants(person):
            reference_pattern = re.escape(reference)
            patterns = [
                rf"\b(?:{request_prefix})\b.{{0,140}}\b{deliverable}\b"
                rf".{{0,120}}\bfrom\s+@?{reference_pattern}\b",
                rf"\b{action_prefix}\b.{{0,140}}\b(?:from|by)\s+"
                rf"@?{reference_pattern}\b",
            ]
            if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
                matches.append(
                    (
                        person,
                        0.97,
                        ["named_deliverable_source_assignment"],
                    )
                )
                break

    return matches if len(matches) == 1 else []


def _has_strong_owner_evidence(
    matches: list[tuple[AssociatedPerson, float, list[str]]],
) -> bool:
    strong = {
        "subject_matter_owner_signal",
        "beneficiary_or_owner_signal",
        "named_request_actor_signal",
    }
    return bool(matches) and bool(strong & set(matches[0][2]))


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


def _accountable_owner_matches(
    request: EmailExtractionRequest,
    email_task_text: str,
    project: ProjectContext,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    normalized = email_task_text.strip()
    if not normalized:
        return []

    project_people = project.people
    direct_addressee = _direct_addressee(request, project_people)
    generic_mention_matches = _generic_request_at_mention_matches(
        request,
        normalized,
        project_people,
    )
    if generic_mention_matches:
        return generic_mention_matches

    matches: list[tuple[AssociatedPerson, float, list[str]]] = []
    for person in project_people:
        if not person.email:
            continue
        if _is_context_reviewer_or_delivery_target_only(
            normalized,
            person,
            include_coordination_targets=False,
        ):
            continue
        evidence: list[str] = []
        score = 0.0

        if _has_subject_matter_owner_signal(normalized, person):
            score = max(score, 0.91)
            evidence.append("subject_matter_owner_signal")
        if _has_beneficiary_or_owner_signal(normalized, person):
            score = max(score, 0.90)
            evidence.append("beneficiary_or_owner_signal")
        if _has_named_request_actor_signal(normalized, person):
            score = max(score, 0.89)
            evidence.append("named_request_actor_signal")
        if person.email == (request.from_participant.email if request.from_participant else ""):
            if _has_sender_self_ownership_signal(normalized):
                score = max(score, 0.91)
                evidence.append("sender_self_ownership_signal")
            if _has_sender_generic_owner_signal(request, normalized, direct_addressee):
                score = max(score, 0.82)
                evidence.append("sender_generic_request_owner_signal")

        if score:
            if direct_addressee and person.email == direct_addressee.email:
                score += 0.02
                evidence.append("direct_addressee_accountable_owner")
            matches.append((person, round(min(1.0, score), 3), sorted(set(evidence))))

    if not matches:
        single_to_match = _single_to_requested_outcome_match(
            request, normalized, direct_addressee, project
        )
        if single_to_match:
            return single_to_match
        return _generic_request_fallback(request, normalized, direct_addressee, project)

    matches.sort(key=lambda item: item[1], reverse=True)
    strong_non_sender_matches = [
        match
        for match in matches
        if match[0].email != (request.from_participant.email if request.from_participant else "")
        and set(match[2])
        & {
            "subject_matter_owner_signal",
            "beneficiary_or_owner_signal",
            "named_request_actor_signal",
        }
    ]
    if strong_non_sender_matches:
        matches = strong_non_sender_matches
    else:
        single_to_match = _single_to_requested_outcome_match(
            request, normalized, direct_addressee, project
        )
        if single_to_match and not _has_sender_self_ownership_signal(normalized):
            return single_to_match

    top_score = matches[0][1]
    selected = [match for match in matches if match[1] >= max(0.84, top_score - 0.03)]
    if selected:
        return selected
    return _generic_request_fallback(request, normalized, direct_addressee, project)


def _direct_addressee(
    request: EmailExtractionRequest,
    project_people: list[AssociatedPerson],
) -> AssociatedPerson | None:
    if not request.visible_recipients:
        return None
    salutation_match = re.match(
        r"^\s*(?:hi|hello|hey)?\s*([a-z][a-z'-]+)\b\s*,",
        request.body_text.strip(),
        re.IGNORECASE,
    )
    if not salutation_match:
        return None
    salutation = salutation_match.group(1).lower()
    recipient_emails = {recipient.email for recipient in request.visible_recipients}
    for person in project_people:
        if person.email not in recipient_emails:
            continue
        if salutation in _person_reference_variants(person):
            return person
    return None


def _has_generic_request_phrase(text: str) -> bool:
    return bool(GENERIC_REQUEST_RE.search(text))


def _has_actionable_outcome(text: str) -> bool:
    if PURE_KNOWLEDGE_RE.search(text) and not ACTIONABLE_OUTCOME_RE.search(text):
        return False
    return bool(ACTIONABLE_OUTCOME_RE.search(text))


def _generic_request_at_mention_matches(
    request: EmailExtractionRequest,
    normalized: str,
    project_people: list[AssociatedPerson],
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if not _has_generic_request_phrase(normalized):
        return []

    raw_text = request.combined_text or normalized
    mentioned: dict[str, AssociatedPerson] = {}
    for match in GENERIC_REQUEST_RE.finditer(raw_text):
        window = raw_text[max(0, match.start() - 180) : match.end() + 220]
        for person in project_people:
            if person.email and _window_mentions_person(window, person):
                mentioned[person.email] = person

    if len(mentioned) == 1:
        person = next(iter(mentioned.values()))
        return [
            (
                person,
                0.94,
                ["same_block_generic_request_at_mention"],
            )
        ]
    return []


def _generic_request_fallback(
    request: EmailExtractionRequest,
    normalized: str,
    direct_addressee: AssociatedPerson | None,
    project: ProjectContext,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if not _has_generic_request_phrase(normalized):
        return []
    if not _has_actionable_outcome(normalized):
        return []
    if re.search(r"\bhelp\s+(?:him|her|them|[a-z][a-z'-]+)\b", normalized, re.IGNORECASE):
        return []

    single_to_person = _single_addressed_project_to_recipient(request, project.people)
    if single_to_person and (
        direct_addressee is None or direct_addressee.email == single_to_person.email
    ):
        return [
            (
                single_to_person,
                0.88,
                [
                    "single_to_generic_request_assignment",
                    "accountable_recipient_can_someone_fallback",
                ],
            )
        ]

    manager = _single_project_manager_in_to(request, project.associated_managers)
    if manager:
        return [
            (
                manager,
                0.86,
                ["project_manager_generic_request_fallback"],
            )
        ]

    return []


def _single_to_requested_outcome_match(
    request: EmailExtractionRequest,
    normalized: str,
    direct_addressee: AssociatedPerson | None,
    project: ProjectContext,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if len(request.to) != 1:
        return []
    if _has_sender_self_ownership_signal(normalized):
        return []
    if _has_generic_request_phrase(normalized):
        return []
    if re.search(r"\b(?:can|could|would)\s+you\b", normalized, re.IGNORECASE):
        return []

    recipient = _single_addressed_project_to_recipient(request, project.people)
    if not recipient:
        return []
    sender_email = request.from_participant.email if request.from_participant else ""
    if recipient.email == sender_email:
        return []
    if direct_addressee and direct_addressee.email != recipient.email:
        return []
    if not _has_requested_outcome_signal(normalized):
        return []

    return [
        (
            recipient,
            0.89,
            ["single_to_requested_outcome_assignment"],
        )
    ]


def _window_mentions_person(window: str, person: AssociatedPerson) -> bool:
    for reference in _person_reference_variants(person):
        if "@" in reference:
            continue
        mention_pattern = rf"(?<!\w)@{re.escape(reference)}\b"
        if re.search(mention_pattern, window, re.IGNORECASE):
            return True
    return False


def _single_addressed_project_to_recipient(
    request: EmailExtractionRequest,
    project_people: list[AssociatedPerson],
) -> AssociatedPerson | None:
    if len(request.to) != 1:
        return None
    recipient_email = request.to[0].email
    for person in project_people:
        if person.email == recipient_email:
            return person
    return None


def _single_project_manager_in_to(
    request: EmailExtractionRequest,
    project_managers: list[AssociatedPerson],
) -> AssociatedPerson | None:
    to_emails = {recipient.email for recipient in request.to}
    managers_in_to = [
        manager for manager in project_managers if manager.email and manager.email in to_emails
    ]
    if len(managers_in_to) == 1:
        return managers_in_to[0]
    return None


def _directional_you_matches(
    request: EmailExtractionRequest,
    normalized: str,
    project_people: list[AssociatedPerson],
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if not re.search(
        r"\b(?:can|could|would)\s+you\b|\bplease\b|\bneed\s+you\s+to\b",
        normalized,
        re.IGNORECASE,
    ):
        return []

    project_people_by_email = {person.email: person for person in project_people if person.email}
    if request.direction == "sent":
        visible_project_recipients = [
            project_people_by_email[recipient.email]
            for recipient in request.visible_recipients
            if recipient.email in project_people_by_email
        ]
        if len(visible_project_recipients) == 1:
            return [
                (
                    visible_project_recipients[0],
                    0.93,
                    ["sent_direction_you_assignment"],
                )
            ]

    if request.direction == "received":
        direct_addressee = _direct_addressee(request, project_people)
        if direct_addressee:
            return [
                (
                    direct_addressee,
                    0.91,
                    ["received_direction_you_accountable_recipient"],
                )
            ]
        mailbox_person = project_people_by_email.get(request.mailbox)
        if mailbox_person:
            return [
                (
                    mailbox_person,
                    0.89,
                    ["received_direction_mailbox_accountable_recipient"],
                )
            ]
    return []


def _has_subject_matter_owner_signal(text: str, person: AssociatedPerson) -> bool:
    for reference in _person_reference_variants(person):
        pattern = (
            rf"\b{re.escape(reference)}(?:'s)?\b"
            r"\s+(?:flagged|noticed|thinks|believes|found|"
            r"identified|is\s+working|has\s+the\s+list|would\s+like\s+to|"
            r"still\s+owes|owes)\b"
        )
        reverse_pattern = (
            r"\b(?:flagged|noticed|thinks|believes|found|identified)\b"
            rf".{0, 40}\b{re.escape(reference)}(?:'s)?\b"
        )
        if re.search(pattern, text, re.IGNORECASE) or re.search(
            reverse_pattern,
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def _has_beneficiary_or_owner_signal(text: str, person: AssociatedPerson) -> bool:
    for reference in _person_reference_variants(person):
        reference_pattern = re.escape(reference)
        patterns = [
            rf"\bhelp\s+{reference_pattern}\b",
            rf"\bmake\s+sure\s+{reference_pattern}\b",
            rf"\bconfirm\s+with\s+{reference_pattern}\b",
            rf"\bbrief\s+{reference_pattern}\b",
            rf"\bwalk\s+{reference_pattern}\b",
            rf"\bresend\b.{0, 60}\bto\s+{reference_pattern}\b",
            rf"\bfor\s+{reference_pattern}\b.{0, 80}\b(?:review|approval|sign[- ]off|use)\b",
            rf"\b{reference_pattern}\b.{0, 80}\b(?:has|needs|is\s+missing|is\s+working\s+from)\b",
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return True

    if _pronoun_owner_signal(text, person):
        return True
    return False


def _has_named_request_actor_signal(text: str, person: AssociatedPerson) -> bool:
    for reference in _person_reference_variants(person):
        reference_pattern = re.escape(reference)
        immediate_pattern = (
            rf"\b(?:can|could|should|will)\s+{reference_pattern}\s+"
            r"(?:confirm|check|review|update|send|handle|coordinate|reconcile|resolve|"
            r"brief|walk|find|complete)\b"
        )
        loose_pattern = (
            rf"\b(?:can|could|should|will)\s+{reference_pattern}\b.{0, 80}"
            r"\b(?:confirm|check|review|update|send|handle|coordinate|reconcile|resolve|"
            r"brief|walk|find|complete)\b"
        )
        if re.search(immediate_pattern, text, re.IGNORECASE) or re.search(
            loose_pattern,
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def _is_context_reviewer_or_delivery_target_only(
    text: str,
    person: AssociatedPerson,
    *,
    include_coordination_targets: bool = True,
) -> bool:
    for reference in _person_reference_variants(person):
        reference_pattern = re.escape(reference)
        target_patterns = [
            rf"\b(?:send|get|provide|share|forward|route|deliver|resend)\b.{0, 100}"
            rf"\b(?:to|for)\s+{reference_pattern}\b",
            rf"\b(?:send|get|provide|share|forward|route|deliver|resend)\b.{0, 100}"
            rf"\bwith\s+{reference_pattern}\b",
            rf"\bfor\s+{reference_pattern}\b.{0, 80}\b(?:review|approval|awareness|records|use)\b",
            rf"\b(?:wants?|would\s+like|asked|asks|needs?)\s+{reference_pattern}\s+to\s+review\b",
            rf"\b{reference_pattern}\b.{0, 60}\b"
            rf"(?:wants?|would\s+like|asked|asks|needs?)\s+to\s+review\b",
            rf"\b{reference_pattern}\b\s+mentioned\b(?![^.?!]{{0,120}}\b(?:he|she|they|him|her|them|working\s+from|owns?|owes|needs|is\s+missing|has\s+the\s+list)\b)",
        ]
        if include_coordination_targets:
            target_patterns.append(
                rf"\b(?:confirm|coordinate|check|follow\s+up)\s+with\s+{reference_pattern}\b"
            )
        actor_patterns = [
            rf"\b{reference_pattern}\b.{0, 40}\b(?:can|could|should|will|needs?\s+to|"
            r"update|prepare|send|coordinate|confirm|resolve|complete)\b",
            rf"\b(?:have|ask|tell|need)\s+{reference_pattern}\b",
            rf"\b(?:have|ask|tell|need)\s+{reference_pattern}\s+to\s+(?!review\b)",
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in target_patterns) and not any(
            re.search(pattern, text, re.IGNORECASE) for pattern in actor_patterns
        ):
            return True
    return False


def _has_sender_self_ownership_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:"
            r"(?:(?:i|we)\s+will|(?:i|we)['’]ll)\s+"
            r"(?:analy[sz]e|complete|create|deliver|draft|handle|own|prepare|"
            r"produce|provide|review|send|summari[sz]e|take|update)|"
            r"i\s+can\s+(?:handle|own|take)|"
            r"we\s+can\s+(?:handle|own|take)|"
            r"i\s+am\s+(?:working\s+on|going\s+to\s+(?:handle|own|take))|"
            r"we\s+are\s+(?:working\s+on|going\s+to\s+(?:handle|own|take))"
            r")\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_requested_outcome_signal(text: str) -> bool:
    return bool(
        ACTIONABLE_OUTCOME_RE.search(text)
        or re.search(
            r"\b(?:need|needs|want|wants|would\s+like|looking\s+for|"
            r"this\s+needs\s+to\s+be|open\s+item|remaining\s+gap|"
            r"still\s+outstanding|at\s+risk\s+of\s+slipping)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_sender_generic_owner_signal(
    request: EmailExtractionRequest,
    text: str,
    direct_addressee: AssociatedPerson | None,
) -> bool:
    if not request.from_participant or not request.from_participant.email:
        return False
    if not _is_internal_tenant_sender(request):
        return False
    if not _has_generic_request_phrase(text) or not _has_actionable_outcome(text):
        return False
    if _has_explicit_accountable_recipient_wording(request, text, direct_addressee):
        return False
    return True


def _is_internal_tenant_sender(request: EmailExtractionRequest) -> bool:
    sender_email = request.from_participant.email if request.from_participant else ""
    sender_domain = sender_email.rsplit("@", 1)[1].lower() if "@" in sender_email else ""
    mailbox_domain = request.mailbox.rsplit("@", 1)[1].lower() if "@" in request.mailbox else ""
    if not sender_domain:
        return False
    tenant_domains = {domain for domain in [mailbox_domain, "taslow.com", "acme.com"] if domain}
    return sender_domain in tenant_domains


def _has_explicit_accountable_recipient_wording(
    request: EmailExtractionRequest,
    text: str,
    direct_addressee: AssociatedPerson | None,
) -> bool:
    if not direct_addressee:
        return False
    salutation = re.match(
        r"^\s*(?:hi|hello|hey)?\s*([a-z][a-z'-]+)\b\s*,",
        request.body_text.strip(),
        re.IGNORECASE,
    )
    if not salutation:
        return False
    return bool(
        re.search(
            r"^\s*(?:hi|hello|hey)?\s*[a-z][a-z'-]+\b\s*,"
            r".{0,120}\b(?:can|could|would)\s+you\b|"
            r"^\s*(?:hi|hello|hey)?\s*[a-z][a-z'-]+\b\s*,"
            r".{0,120}\b(?:please|need\s+you\s+to)\b",
            text,
            re.IGNORECASE,
        )
    )


def _pronoun_owner_signal(text: str, person: AssociatedPerson) -> bool:
    for reference in _person_reference_variants(person):
        first_name = reference.split()[0]
        if not first_name or len(first_name) <= 2:
            continue
        named_then_pronoun = (
            rf"\b{re.escape(first_name)}(?:'s)?\b.{0, 120}"
            r"\b(?:help\s+(?:him|her|them)|make\s+sure\s+(?:he|she|they)|"
            r"confirm\s+with\s+(?:him|her|them)|(?:he|she|they)(?:'s|\s+is|\s+are)?"
            r"\s+working\s+from|(?:he|she|they)\s+(?:mentioned|flagged|found|identified))\b"
        )
        if re.search(named_then_pronoun, text, re.IGNORECASE):
            return True
    return False
