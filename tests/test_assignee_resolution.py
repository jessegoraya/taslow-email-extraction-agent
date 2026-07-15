from __future__ import annotations

from taslow_email_extraction_agent.executors.assignee_resolution import resolve_assignees
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
)


async def test_subject_matter_owner_beats_single_visible_recipient():
    request = _request(
        body=(
            "Jesse, David flagged that the subcontractor invoice includes an equipment "
            "rental line that does not match the approved change order log. David wants "
            "to sort it out before approving payment."
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[{"email": "david@tenant.com", "name": "David"}],
    )
    task = _task("Resolve the subcontractor invoice discrepancy before approving payment.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "subject_matter_owner_signal" in matches[0][2]


async def test_pronoun_help_signal_prefers_named_project_owner():
    request = _request(
        body=(
            "Jesse, David's still working through the load-bank test delay impact. "
            "Can someone help him get that assessment done soon?"
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Help David finish the load-bank delay impact assessment.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "beneficiary_or_owner_signal" in matches[0][2]


async def test_can_someone_without_named_owner_falls_back_to_accountable_recipient():
    request = _request(
        body="Jesse, can someone update the electrical notes before the review?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Update the electrical notes before the review.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "accountable_recipient_can_someone_fallback" in matches[0][2]


async def test_generic_request_phrase_variant_falls_back_to_single_to_recipient():
    request = _request(
        body="Jesse, would anyone be willing to review the electrical notes before the review?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Review the electrical notes before the review.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "single_to_generic_request_assignment" in matches[0][2]


async def test_generic_request_same_block_at_mention_beats_recipient_fallback():
    request = _request(
        body=(
            "Can someone review the logistics handoff notes before the supplier "
            "call? @Jason Jesse and David are copied for awareness."
        ),
        to=[
            {"email": "jesse@tenant.com", "name": "Jesse"},
            {"email": "david@tenant.com", "name": "David"},
        ],
        cc=[],
    )
    task = _task("Review the logistics handoff notes before the supplier call.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jason@tenant.com"
    assert "same_block_generic_request_at_mention" in matches[0][2]


async def test_generic_request_multiple_to_falls_back_to_single_project_manager():
    request = _request(
        body="Could someone please update the stakeholder deck before Friday?",
        to=[
            {"email": "jesse@tenant.com", "name": "Jesse"},
            {"email": "david@tenant.com", "name": "David"},
        ],
        cc=[],
    )
    task = _task("Update the stakeholder deck before Friday.")

    matches = await resolve_assignees(request, task, _project_with_manager())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "project_manager_generic_request_fallback" in matches[0][2]


async def test_generic_request_multiple_project_managers_remains_unresolved():
    request = _request(
        body="Could someone please update the stakeholder deck before Friday?",
        to=[
            {"email": "jesse@tenant.com", "name": "Jesse"},
            {"email": "maria@tenant.com", "name": "Maria"},
            {"email": "david@tenant.com", "name": "David"},
        ],
        cc=[],
    )
    task = _task("Update the stakeholder deck before Friday.")

    matches = await resolve_assignees(request, task, _project_with_two_managers())

    assert matches == []


async def test_generic_knowledge_question_without_actionable_work_is_unresolved():
    request = _request(
        body="Jesse, does anyone know whether this was already handled?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Determine whether this was already handled.")

    matches = await resolve_assignees(request, task, _project())

    assert matches == []


async def test_single_to_recipient_owns_generic_request_over_project_sender():
    request = _request(
        body="Could someone please confirm the CBA rate variance before we close the audit?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        from_email="david@tenant.com",
    )
    task = _task("Confirm the CBA rate variance before closing the audit.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "single_to_generic_request_assignment" in matches[0][2]


async def test_external_sender_generic_request_assigns_addressed_project_recipient():
    request = _request(
        body=(
            "Jesse, thanks for the draft diagrams. Can someone update the data flow "
            "diagram and resend by Thursday?"
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        from_email="client@example.gov",
    )
    task = _task("Update the data flow diagram and resend by Thursday.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "sender_generic_request_owner_signal" not in matches[0][2]


async def test_project_sender_does_not_override_explicit_recipient_command():
    request = _request(
        body="Jesse, can you confirm the CBA rate variance before we close the audit?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        from_email="david@tenant.com",
    )
    task = _task("Confirm the CBA rate variance before closing the audit.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "sender_generic_request_owner_signal" not in matches[0][2]


async def test_direct_address_in_body_beats_model_task_without_name():
    request = _request(
        body="Sophia, can you update the tracking sheet by June 17?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[{"email": "sophia@tenant.com", "name": "Sophia"}],
    )
    task = _task("Update the tracking sheet by June 17.")

    matches = await resolve_assignees(request, task, _project_with_sophia())

    assert matches[0][0].email == "sophia@tenant.com"
    assert "direct_address_assignment" in matches[0][2]


async def test_delegated_actor_in_body_beats_accountable_recipient():
    request = _request(
        body="Jesse, can you have Jason send the latest process maps?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Send the latest process maps.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jason@tenant.com"
    assert "delegated_assignment_language" in matches[0][2]


async def test_delivery_target_does_not_become_extra_assignee():
    request = _request(
        body=(
            "Mary, can you draft a noise mitigation plan for the renovation and send "
            "it to Elizabeth by June 17?"
        ),
        to=[{"email": "mary@tenant.com", "name": "Mary"}],
        cc=[{"email": "elizabeth@tenant.com", "name": "Elizabeth"}],
    )
    task = _task("Draft a noise mitigation plan and send it to Elizabeth by June 17.")

    matches = await resolve_assignees(request, task, _project_with_delivery_target())

    emails = [match[0].email for match in matches]
    assert emails == ["mary@tenant.com"]


async def test_reviewer_target_does_not_become_extra_assignee():
    request = _request(
        body=(
            "Mary, can you share the closeout binder with Elizabeth? "
            "Krista wants Elizabeth to review it before Friday."
        ),
        to=[{"email": "mary@tenant.com", "name": "Mary"}],
        cc=[{"email": "elizabeth@tenant.com", "name": "Elizabeth"}],
    )
    task = _task("Share the closeout binder with Elizabeth for review before Friday.")

    matches = await resolve_assignees(request, task, _project_with_delivery_target())

    emails = [match[0].email for match in matches]
    assert emails == ["mary@tenant.com"]


async def test_share_with_target_does_not_become_assignee_from_recipient_overlap():
    request = _request(
        body="Mary, please share the latest tracker with Elizabeth before the review.",
        to=[
            {"email": "mary@tenant.com", "name": "Mary"},
            {"email": "elizabeth@tenant.com", "name": "Elizabeth"},
        ],
        cc=[],
    )
    task = _task("Share the latest tracker with Elizabeth before the review.")

    matches = await resolve_assignees(request, task, _project_with_delivery_target())

    emails = [match[0].email for match in matches]
    assert emails == ["mary@tenant.com"]


async def test_mentioned_context_person_does_not_become_assignee():
    request = _request(
        body="Please update the tracker David mentioned before Friday.",
        to=[
            {"email": "mary@tenant.com", "name": "Mary"},
            {"email": "david@tenant.com", "name": "David"},
        ],
        cc=[],
    )
    task = _task("Update the tracker David mentioned before Friday.")

    matches = await resolve_assignees(request, task, _project_with_mentioned_context())

    emails = [match[0].email for match in matches]
    assert "david@tenant.com" not in emails


async def test_pronoun_antecedent_resolves_back_to_named_project_person():
    request = _request(
        body=(
            "Jesse, David mentioned he's working from the old CBA cycle sheet. "
            "Could anyone tell me if we can confirm with him before closing the audit?"
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Confirm with David before closing the audit.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"


async def test_modal_named_actor_is_selected_from_project_people():
    request = _request(
        body=(
            "Jesse, this is the training supplier. Can Jason confirm receipt of last "
            "week's equipment shipment so we can close the PO?"
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
    )
    task = _task("Confirm receipt of last week's equipment shipment.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jason@tenant.com"
    assert set(matches[0][2]) & {"named_request_actor_signal", "modal_named_actor_assignment"}


async def test_sent_direction_you_assigns_single_project_recipient():
    request = _request(
        body="Can you update the closeout binder before the review?",
        to=[{"email": "david@tenant.com", "name": "David"}],
        cc=[],
        direction="sent",
        from_email="jesse@tenant.com",
    )
    task = _task("Update the closeout binder before the review.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "sent_direction_you_assignment" in matches[0][2]


async def test_received_direction_you_assigns_addressed_mailbox_user():
    request = _request(
        body="Jesse, can you draft the one-page status summary by Wednesday?",
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        direction="received",
    )
    task = _task("Draft the one-page status summary by Wednesday.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert set(matches[0][2]) & {
        "received_direction_you_accountable_recipient",
        "direct_address_assignment",
    }


async def test_still_owes_signal_beats_received_direction_you():
    request = _request(
        body=(
            "Jesse, David still owes us the closeout binder notes. "
            "Can you make sure those get out by end of week?"
        ),
        to=[{"email": "jesse@tenant.com", "name": "Jesse"}],
        cc=[],
        direction="received",
    )
    task = _task("Send out the closeout binder notes by end of week.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "subject_matter_owner_signal" in matches[0][2]


async def test_at_mention_yours_to_drive_is_explicit_assignment():
    request = _request(
        body=(
            "@David, this is yours to drive. The closeout binder gap is still outstanding "
            "for Friday's package."
        ),
        to=[
            {"email": "jesse@tenant.com", "name": "Jesse"},
            {"email": "david@tenant.com", "name": "David"},
        ],
        cc=[],
        direction="received",
    )
    task = _task("Resolve the closeout binder gap for Friday's package.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "named_owner_drive_assignment" in matches[0][2]


async def test_single_to_recipient_owns_requested_outcome_not_sender():
    request = _request(
        body="I need the warranty claim process set up by 4/14.",
        to=[{"email": "david@tenant.com", "name": "David"}],
        cc=[],
        direction="received",
        from_email="jesse@tenant.com",
    )
    task = _task("Set up the warranty claim process by 4/14.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "david@tenant.com"
    assert "single_to_requested_outcome_assignment" in matches[0][2]


async def test_sender_self_ownership_beats_single_to_recipient():
    request = _request(
        body="I'll handle the warranty claim process setup by 4/14.",
        to=[{"email": "david@tenant.com", "name": "David"}],
        cc=[],
        direction="received",
        from_email="jesse@tenant.com",
    )
    task = _task("Handle the warranty claim process setup by 4/14.")

    matches = await resolve_assignees(request, task, _project())

    assert matches[0][0].email == "jesse@tenant.com"
    assert "sender_self_ownership_signal" in matches[0][2]


def _project() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", role="Manager"),
            AssociatedPerson(name="David Vance", email="david@tenant.com", role="Lead"),
            AssociatedPerson(name="Jason Talbot", email="jason@tenant.com", role="Logistics"),
        ],
        associatedManagers=[],
        scopes=[],
    )


def _project_with_manager() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="David Vance", email="david@tenant.com", role="Lead"),
            AssociatedPerson(name="Jason Talbot", email="jason@tenant.com", role="Logistics"),
        ],
        associatedManagers=[
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", role="Manager")
        ],
        scopes=[],
    )


def _project_with_sophia() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", role="Manager"),
            AssociatedPerson(name="Sophia Green", email="sophia@tenant.com", role="Lead"),
        ],
        associatedManagers=[],
        scopes=[],
    )


def _project_with_delivery_target() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="Mary Ann", email="mary@tenant.com", role="Lead"),
            AssociatedPerson(
                name="Elizabeth Romero",
                email="elizabeth@tenant.com",
                role="Reviewer",
            ),
        ],
        associatedManagers=[],
        scopes=[],
    )


def _project_with_mentioned_context() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="Mary Ann", email="mary@tenant.com", role="Lead"),
            AssociatedPerson(name="David Vance", email="david@tenant.com", role="Reviewer"),
        ],
        associatedManagers=[],
        scopes=[],
    )


def _project_with_two_managers() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Lake Nona",
        description="Construction project",
        associatedPeople=[
            AssociatedPerson(name="David Vance", email="david@tenant.com", role="Lead"),
            AssociatedPerson(name="Jason Talbot", email="jason@tenant.com", role="Logistics"),
        ],
        associatedManagers=[
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", role="Manager"),
            AssociatedPerson(name="Maria Ortiz", email="maria@tenant.com", role="Manager"),
        ],
        scopes=[],
    )


def _request(
    body: str,
    to: list[dict[str, str]],
    cc: list[dict[str, str]],
    direction: str = "received",
    from_email: str = "external@example.com",
) -> EmailExtractionRequest:
    return EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction=direction,
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Lake Nona - implicit task",
        bodyText=body,
        **{"from": {"email": from_email, "name": "External"}},
        to=to,
        cc=cc,
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )


def _task(description: str) -> ExtractedTaskCandidate:
    return ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title=description,
        description=description,
        mentionedPeople=[],
        dueText=None,
        confidence=0.9,
        evidence=["implicit_task_language"],
    )
