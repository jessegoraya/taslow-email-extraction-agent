from __future__ import annotations

from taslow_email_extraction_agent.executors.project_matching import (
    _project_alias_match,
    _project_aliases,
    score_project_candidates,
)
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectScope,
)


async def test_external_sender_can_match_project_when_project_person_and_search_agree(project):
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="operations@external.example",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Cube electrical scope",
        bodyText="Please have Tessa update the electrical scope before the readiness review.",
        **{"from": {"email": "inspector@external.example", "name": "External Inspector"}},
        to=[{"email": "tessa@tenant.com", "name": "Tessa"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Update electrical scope",
        description="Have Tessa update the electrical scope before the readiness review.",
        mentionedPeople=["Tessa"],
        dueText=None,
        confidence=0.92,
        evidence=["explicit_task_language"],
    )
    searched_project = project.model_copy(
        update={"search_score": 0.72, "search_rank": 1, "search_margin": 0.08}
    )

    score = await score_project_candidates(
        request,
        [task],
        [searched_project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence >= 0.80
    assert score.result.decision_reason == "explicit_unique_scope_title_reference"
    assert "explicit_unique_scope_title_reference" in score.result.evidence
    assert "external_sender_allowed_with_project_people_context" in score.result.evidence


async def test_equal_confidence_prefers_stronger_project_participant_overlap():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="alex@tenant.example",
        direction="received",
        graphEventId="graph-project-tie",
        internetMessageId="<project-tie@example.com>",
        messageId="project-tie",
        subject="Draft analysis follow-up",
        bodyText="Please prepare the draft analysis for the next review.",
        **{"from": {"email": "client@external.example", "name": "External Client"}},
        to=[{"email": "alex@tenant.example", "name": "Alex"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-project-tie",
        correlationId="corr-project-tie",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Prepare the draft analysis",
        description="Prepare the draft analysis for the next review.",
        mentionedPeople=["Alex"],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    weaker_overlap = _project(
        project_id="search-rank-one",
        name="Operations Planning Support",
        search_score=0.72,
        search_rank=1,
        search_margin=0.01,
        people=[AssociatedPerson(name="Alex", email="alex@tenant.example")],
    )
    stronger_overlap = _project(
        project_id="participant-match",
        name="Program Review Support",
        search_score=0.69,
        search_rank=7,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Alex", email="alex@tenant.example"),
            AssociatedPerson(name="External Client", email="client@external.example"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [weaker_overlap, stronger_overlap],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "participant-match"
    assert score.result.confidence < 0.80


async def test_unique_sender_recipient_project_beats_sender_only_semantic_candidate():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="paco@bloomsky.onmicrosoft.com",
        direction="received",
        graphEventId="graph-shared-project",
        internetMessageId="<shared-project@bloomsky.onmicrosoft.com>",
        messageId="shared-project",
        subject="Draft status package",
        bodyText="Paco, please prepare the draft status package for Friday's review.",
        **{"from": {"email": "alex@bloomsky.onmicrosoft.com", "name": "Alex"}},
        to=[{"email": "paco@bloomsky.onmicrosoft.com", "name": "Paco"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-shared-project",
        correlationId="corr-shared-project",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Prepare the draft status package",
        description="Prepare the draft status package for Friday's review.",
        mentionedPeople=["Paco"],
        confidence=0.93,
        evidence=["explicit_task_language"],
    )
    education = _project(
        project_id="education",
        name="Education Wwc Pesto",
        search_score=0.92,
        search_rank=1,
        search_margin=0.18,
        people=[AssociatedPerson(name="Alex", email="alex@bloomsky.onmicrosoft.com")],
    )
    va_radiology = _project(
        project_id="va-radiology",
        name="VA Radiology Staffing and Medical Support",
        search_score=0.0,
        search_rank=0,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Alex", email="alex@bloomsky.onmicrosoft.com"),
            AssociatedPerson(name="Paco", email="paco@bloomsky.onmicrosoft.com"),
        ],
    ).model_copy(update={"candidate_sources": ["participant_overlap"]})

    score = await score_project_candidates(
        request,
        [task],
        [education, va_radiology],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "va-radiology"
    assert score.result.confidence >= 0.80
    assert score.result.decision_reason == "unique_sender_recipient_project_overlap"
    assert score.result.participant_overlap_count == 2
    assert score.result.recipient_overlap_count == 1
    assert score.result.sender_project_member is True
    assert "participant_project_candidate_expansion" in score.result.evidence


async def test_highest_multi_participant_overlap_precedes_semantic_tiebreak():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="paco@tenant.example",
        direction="received",
        graphEventId="graph-multi-overlap",
        internetMessageId="<multi-overlap@tenant.example>",
        messageId="multi-overlap",
        subject="Review package",
        bodyText="Paco and Tessa, please finish the review package by Friday.",
        **{"from": {"email": "alex@tenant.example", "name": "Alex"}},
        to=[
            {"email": "paco@tenant.example", "name": "Paco"},
            {"email": "tessa@tenant.example", "name": "Tessa"},
        ],
        cc=[],
        bcc=[],
        idempotencyKey="key-multi-overlap",
        correlationId="corr-multi-overlap",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Finish the review package",
        description="Paco and Tessa should finish the review package by Friday.",
        mentionedPeople=["Paco", "Tessa"],
        confidence=0.93,
        evidence=["explicit_task_language"],
    )
    two_person_project = _project(
        project_id="two-person",
        name="Semantic Review Support",
        search_score=0.91,
        search_rank=1,
        search_margin=0.15,
        people=[
            AssociatedPerson(name="Alex", email="alex@tenant.example"),
            AssociatedPerson(name="Paco", email="paco@tenant.example"),
        ],
    )
    three_person_project = _project(
        project_id="three-person",
        name="Joint Review Support",
        search_score=0.62,
        search_rank=5,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Alex", email="alex@tenant.example"),
            AssociatedPerson(name="Paco", email="paco@tenant.example"),
            AssociatedPerson(name="Tessa", email="tessa@tenant.example"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [two_person_project, three_person_project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "three-person"
    assert score.result.decision_reason == "strongest_participant_project_overlap"
    assert score.result.participant_overlap_count == 3


async def test_tied_sender_recipient_overlap_uses_semantic_context():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="paco@tenant.example",
        direction="received",
        graphEventId="graph-overlap-tie",
        internetMessageId="<overlap-tie@tenant.example>",
        messageId="overlap-tie",
        subject="Radiology staffing package",
        bodyText="Paco, please prepare the radiology staffing package by Friday.",
        **{"from": {"email": "alex@tenant.example", "name": "Alex"}},
        to=[{"email": "paco@tenant.example", "name": "Paco"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-overlap-tie",
        correlationId="corr-overlap-tie",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="task-1",
        title="Prepare the radiology staffing package",
        description="Prepare the radiology staffing package by Friday.",
        mentionedPeople=["Paco"],
        confidence=0.93,
        evidence=["explicit_task_language"],
    )
    semantically_expected = _project(
        project_id="radiology",
        name="Radiology Staffing Support",
        search_score=0.86,
        search_rank=1,
        search_margin=0.18,
        people=[
            AssociatedPerson(name="Alex", email="alex@tenant.example"),
            AssociatedPerson(name="Paco", email="paco@tenant.example"),
        ],
    )
    weaker_context = _project(
        project_id="training",
        name="Training Support",
        search_score=0.64,
        search_rank=3,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Alex", email="alex@tenant.example"),
            AssociatedPerson(name="Paco", email="paco@tenant.example"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [weaker_context, semantically_expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "radiology"
    assert score.result.participant_overlap_count == 2
    assert score.result.decision_reason == "strong_search_and_participant_evidence"


async def test_participants_alone_do_not_match_an_unknown_named_workstream():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="capri@tenant.example",
        direction="received",
        graphEventId="graph-no-project",
        internetMessageId="<no-project@example.com>",
        messageId="no-project",
        subject="Blue Lantern Vendor Transition",
        bodyText=(
            "Capri should validate the Blue Lantern Vendor Transition update by "
            "2026-08-06. Keep that workstream open until the update is checked."
        ),
        **{"from": {"email": "david@tenant.example", "name": "David"}},
        to=[{"email": "capri@tenant.example", "name": "Capri"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-no-project",
        correlationId="corr-no-project",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Validate the Blue Lantern Vendor Transition update",
        description="Validate the Blue Lantern Vendor Transition update by 2026-08-06.",
        mentionedPeople=["Capri"],
        dueText="2026-08-06",
        confidence=0.94,
        evidence=["explicit_task_language"],
    )
    unrelated = _project(
        project_id="unrelated-project",
        name="Radiology Staffing and Medical Support",
        search_score=0.70,
        search_rank=4,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Capri", email="capri@tenant.example"),
            AssociatedPerson(name="David", email="david@tenant.example"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [unrelated],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence < 0.80
    assert score.result.decision_reason == "weighted_evidence"


async def test_exact_generic_scope_and_participants_can_match_project():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="sofia@tenant.example",
        direction="received",
        graphEventId="graph-generic-scope-route",
        internetMessageId="<generic-scope-route@example.com>",
        messageId="generic-scope-route",
        subject="Contract Administration And Management review",
        bodyText=(
            "Sofia, please confirm the Contract Administration And Management "
            "analysis by 2026-08-17."
        ),
        **{"from": {"email": "manager@tenant.example", "name": "Manager"}},
        to=[{"email": "sofia@tenant.example", "name": "Sofia"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-generic-scope-route",
        correlationId="corr-generic-scope-route",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Confirm the Contract Administration And Management analysis",
        description="Confirm the Contract Administration And Management analysis.",
        mentionedPeople=["Sofia"],
        dueText="2026-08-17",
        confidence=0.94,
        evidence=["explicit_task_language"],
    )
    expected = _project(
        project_id="pest-control",
        name="Pest Control Services",
        search_score=0.71,
        search_rank=3,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Sofia", email="sofia@tenant.example"),
            AssociatedPerson(name="Manager", email="manager@tenant.example"),
        ],
        scopes=[
            ProjectScope(
                scopeId="contract-admin",
                title="Contract Administration And Management",
                description="Contract administration requirements.",
            )
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence >= 0.80
    assert score.result.decision_reason == "explicit_scope_and_participant_evidence"
    assert "explicit_scope_title_reference" in score.result.evidence


async def test_equal_confidence_and_participants_preserve_search_order():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="alex@tenant.example",
        direction="received",
        graphEventId="graph-project-stable",
        internetMessageId="<project-stable@example.com>",
        messageId="project-stable",
        subject="Draft analysis follow-up",
        bodyText="Please prepare the draft analysis for the next review.",
        **{"from": {"email": "client@external.example", "name": "External Client"}},
        to=[
            {"email": "alex@tenant.example", "name": "Alex"},
            {"email": "beth@tenant.example", "name": "Beth"},
        ],
        cc=[],
        bcc=[],
        idempotencyKey="key-project-stable",
        correlationId="corr-project-stable",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Prepare the draft analysis",
        description="Prepare the draft analysis for the next review.",
        mentionedPeople=["Alex"],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    search_rank_one = _project(
        project_id="search-rank-one",
        name="Operations Planning Support",
        search_score=0.72,
        search_rank=1,
        search_margin=0.01,
        people=[AssociatedPerson(name="Alex", email="alex@tenant.example")],
    )
    later_candidate = _project(
        project_id="later-candidate",
        name="Program Review Support",
        search_score=0.69,
        search_rank=7,
        search_margin=0.0,
        people=[AssociatedPerson(name="Alex", email="alex@tenant.example")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [search_rank_one, later_candidate],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "search-rank-one"
    assert "project_selection_participant_tiebreak" not in score.result.evidence


async def test_equal_primary_evidence_uses_weighted_project_evidence():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="alex@tenant.example",
        direction="received",
        graphEventId="graph-project-weighted-tie",
        internetMessageId="<project-weighted-tie@example.com>",
        messageId="project-weighted-tie",
        subject="Personnel training analysis",
        bodyText=(
            "Please prepare the personnel hiring and training analysis for distribution support."
        ),
        **{"from": {"email": "client@external.example", "name": "External Client"}},
        to=[{"email": "alex@tenant.example", "name": "Alex"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-project-weighted-tie",
        correlationId="corr-project-weighted-tie",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Prepare personnel hiring and training analysis",
        description=(
            "Prepare the personnel hiring and training analysis for distribution support."
        ),
        mentionedPeople=["Alex"],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    search_rank_one = _project(
        project_id="internship",
        name="Internship Program Support",
        search_score=0.72,
        search_rank=1,
        search_margin=0.01,
        people=[AssociatedPerson(name="Alex", email="alex@tenant.example")],
    )
    stronger_combined_evidence = _project(
        project_id="distribution",
        name="Distribution Personnel Training Support",
        search_score=0.70,
        search_rank=7,
        search_margin=0.0,
        people=[AssociatedPerson(name="Alex", email="alex@tenant.example")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [search_rank_one, stronger_combined_evidence],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "distribution"
    assert score.result.confidence < 0.80
    assert score.result.decision_reason == "weak_external_project_anchor"


async def test_subject_alias_rank_one_project_beats_weak_external_sender_candidate():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@taslow.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="AES2 - architecture diagram review feedback",
        bodyText=(
            "Jesse, thanks for sending the draft diagrams over. Can someone update the data "
            "flow diagram to reflect the new ingestion path and resend by Thursday?"
        ),
        **{"from": {"email": "reviewer@example.com", "name": "External Reviewer"}},
        to=[{"email": "jesse@taslow.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Update data flow diagram",
        description="Update the data flow diagram to reflect the new ingestion path.",
        mentionedPeople=["Jesse"],
        dueText="by Thursday",
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    expected = _project(
        project_id="aes2",
        name="Architecture Engineering Services 2 (AES2) updated 2",
        search_score=0.7442,
        search_rank=1,
        search_margin=0.04,
        people=[
            AssociatedPerson(name="Jesse", email="jesse@taslow.com"),
            AssociatedPerson(name="Ava", email="ava@taslow.com"),
        ],
    )
    weak_external = _project(
        project_id="badssd",
        name="Business Architecture Development & Strategic Specification Design (BAD&SSD)",
        search_score=0.7036,
        search_rank=2,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="Jesse", email="jesse@taslow.com"),
            AssociatedPerson(name="Krista", email="krista@taslow.com"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [weak_external, expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "aes2"
    assert score.result.decision_reason == "subject_alias_search_and_participant_evidence"
    assert "subject_project_alias_match" in score.result.evidence


async def test_body_alias_supports_rank_one_project_when_subject_is_generic():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@taslow.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Architecture diagram review feedback",
        bodyText=(
            "The AES2 design package still needs the data flow diagram updated before "
            "Thursday's review."
        ),
        **{"from": {"email": "reviewer@example.com", "name": "External Reviewer"}},
        to=[{"email": "jesse@taslow.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Update data flow diagram",
        description="Update the AES2 data flow diagram before Thursday's review.",
        mentionedPeople=["Jesse"],
        dueText="before Thursday",
        confidence=0.90,
        evidence=["implicit_task_language"],
    )
    expected = _project(
        project_id="aes2",
        name="Architecture Engineering Services 2 (AES2) updated 2",
        search_score=0.70,
        search_rank=1,
        search_margin=0.02,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )
    weak_external = _project(
        project_id="badssd",
        name="Business Architecture Development & Strategic Specification Design (BAD&SSD)",
        search_score=0.69,
        search_rank=2,
        search_margin=0.0,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [weak_external, expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "aes2"
    assert "body_project_alias_match" in score.result.evidence


async def test_subject_alias_outweighs_body_alias_for_different_project():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@taslow.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="AES2 - architecture diagram review feedback",
        bodyText=(
            "This may look similar to prior BAD&SSD planning work, but the current "
            "diagram update belongs with the AES2 package."
        ),
        **{"from": {"email": "reviewer@example.com", "name": "External Reviewer"}},
        to=[{"email": "jesse@taslow.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Update architecture diagram",
        description="Update the diagram for the AES2 package.",
        mentionedPeople=["Jesse"],
        dueText=None,
        confidence=0.90,
        evidence=["implicit_task_language"],
    )
    subject_project = _project(
        project_id="aes2",
        name="Architecture Engineering Services 2 (AES2) updated 2",
        search_score=0.70,
        search_rank=1,
        search_margin=0.01,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )
    body_mentioned_project = _project(
        project_id="badssd",
        name="Business Architecture Development & Strategic Specification Design (BAD&SSD)",
        search_score=0.71,
        search_rank=2,
        search_margin=0.0,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [body_mentioned_project, subject_project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "aes2"
    assert "subject_project_alias_match" in score.result.evidence


def test_project_aliases_include_parenthetical_and_generic_suffix_acronym_prefixes():
    assert "AES2" in _project_aliases("Architecture Engineering Services 2 (AES2) updated 2")
    assert "TMC" in _project_aliases("TRAVEL MANAGEMENT CENTER SERVICES")


def test_body_alias_matching_uses_only_high_confidence_aliases():
    badssd = _project(
        project_id="badssd",
        name="Business Architecture Development & Strategic Specification Design (BAD&SSD)",
        search_score=0.72,
        search_rank=1,
        search_margin=0.05,
        people=[],
    )
    inspection = _project(
        project_id="isar",
        name="INSPECTION SERVICES AND REPORTING",
        search_score=0.72,
        search_rank=1,
        search_margin=0.05,
        people=[],
    )

    assert _project_alias_match(
        "BAD&SSD review items are ready.",
        badssd,
        high_confidence_only=True,
    )
    assert not _project_alias_match(
        "That would be a bad look for the vendor review.",
        badssd,
        high_confidence_only=True,
    )
    assert not _project_alias_match(
        "Please prepare the inspection notes for the site visit.",
        inspection,
        high_confidence_only=True,
    )


def test_subject_alias_matching_ignores_common_uppercase_project_words():
    project = _project(
        project_id="lake-nona",
        name=(
            "RENOVATE LAKE NONA BUILDING 2 FOR EMERGENCY DEPARTMENT AND "
            "OBSERVATION UNIT IN ORLANDO FL (675-23-100)(BB)"
        ),
        search_score=0.72,
        search_rank=1,
        search_margin=0.05,
        people=[],
    )

    assert not _project_alias_match(
        "Cascade Health Portal - need updated copy for error messages",
        project,
    )
    assert _project_alias_match("Lake Nona - update observation unit checklist", project)


async def test_weak_external_sender_evidence_is_capped_below_project_threshold():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@taslow.com",
        direction="received",
        graphEventId="graph-1",
        internetMessageId="<msg-1@example.com>",
        messageId="msg-1",
        subject="Website cleanup",
        bodyText=(
            "Jesse, can you ask someone to clean up the strategic specification design notes? "
            "The current wording is a bad look."
        ),
        **{"from": {"email": "vendor@external.example", "name": "External Vendor"}},
        to=[{"email": "jesse@taslow.com", "name": "Jesse"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-weak-external",
        correlationId="corr-weak-external",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Clean up vendor website notes",
        description=(
            "Clean up the strategic specification design notes because the wording is a bad look."
        ),
        mentionedPeople=["Jesse"],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    common_leader_project = _project(
        project_id="badssd",
        name="Business Architecture Development & Strategic Specification Design (BAD&SSD)",
        search_score=0.88,
        search_rank=1,
        search_margin=0.04,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )
    other_project = _project(
        project_id="aes2",
        name="Architecture Engineering Services 2 (AES2) updated 2",
        search_score=0.70,
        search_rank=2,
        search_margin=0.0,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )
    third_project = _project(
        project_id="tmc",
        name="TRAVEL MANAGEMENT CENTER SERVICES",
        search_score=0.68,
        search_rank=3,
        search_margin=0.0,
        people=[AssociatedPerson(name="Jesse", email="jesse@taslow.com")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [common_leader_project, other_project, third_project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence < 0.80
    assert score.result.decision_reason == "weak_external_project_anchor"
    assert "body_project_alias_match" not in score.result.evidence
    assert "weak_external_project_anchor" in score.result.evidence


async def test_unique_external_client_domain_can_anchor_client_originated_task():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="grace@acme-consulting.example",
        direction="received",
        graphEventId="graph-client-domain",
        internetMessageId="<msg-client-domain@example.com>",
        messageId="msg-client-domain",
        subject="Power platform form check",
        bodyText=(
            "Grace, can you verify the diver operations intake form is ready for the next "
            "USCG review cycle?"
        ),
        **{"from": {"email": "client@uscg.mil", "name": "USCG Client"}},
        to=[{"email": "grace@acme-consulting.example", "name": "Grace"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-client-domain",
        correlationId="corr-client-domain",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Verify diver operations intake form",
        description="Verify the diver operations intake form is ready for the next review cycle.",
        mentionedPeople=["Grace"],
        dueText=None,
        confidence=0.90,
        evidence=["explicit_task_language"],
    )
    uscg_project = _project(
        project_id="uscg-power-platform",
        name="USCG Power Platform Modernization and Diver Operations Support",
        search_score=0.58,
        search_rank=1,
        search_margin=0.04,
        people=[AssociatedPerson(name="Grace", email="grace@acme-consulting.example")],
        client_domains=["uscg.mil"],
    )
    unrelated_project = _project(
        project_id="fda-md-sim",
        name="FDA Molecular Modeling Simulation Platform",
        search_score=0.55,
        search_rank=2,
        search_margin=0.0,
        people=[],
        client_domains=["hhs.gov"],
    )

    score = await score_project_candidates(
        request,
        [task],
        [uscg_project, unrelated_project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "uscg-power-platform"
    assert score.result.confidence >= 0.80
    assert score.result.decision_reason == "unique_client_domain_and_search_evidence"
    assert "unique_external_client_domain_match" in score.result.evidence
    assert score.result.client_domain_score == 1.0


async def test_open_item_language_bridges_search_and_participant_project_evidence():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="amanda@acme-consulting.example",
        direction="received",
        graphEventId="graph-open-item",
        internetMessageId="<msg-open-item@example.com>",
        messageId="msg-open-item",
        subject="MD Simulation Platform - open item",
        bodyText=(
            "Amanda, one open item remains around how existing project states will be "
            "preserved during the migration."
        ),
        **{"from": {"email": "lead@acme-consulting.example", "name": "Program Lead"}},
        to=[{"email": "amanda@acme-consulting.example", "name": "Amanda Johnson"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-open-item",
        correlationId="corr-open-item",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Preserve existing project states",
        description="Resolve the open item for preserving existing project states.",
        mentionedPeople=["Amanda"],
        dueText=None,
        confidence=0.90,
        evidence=["implicit_task_language"],
    )
    project = _project(
        project_id="fda-md-sim",
        name="FDA Molecular Modeling Simulation Platform",
        search_score=0.71,
        search_rank=1,
        search_margin=0.03,
        people=[AssociatedPerson(name="Amanda Johnson", email="amanda@acme-consulting.example")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [project],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence >= 0.80
    assert "implicit_open_item_project_signal" in score.result.evidence


async def test_unique_scope_title_reference_overrides_ambiguous_participant_search():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="david@tenant.example",
        direction="received",
        graphEventId="graph-unique-scope",
        internetMessageId="<unique-scope@example.com>",
        messageId="unique-scope",
        subject="Quality Assurance Surveillance Plan review",
        bodyText=(
            "David, please capture final edits to the Quality Assurance Surveillance Plan "
            "before Friday."
        ),
        **{"from": {"email": "lead@external.example", "name": "External Lead"}},
        to=[{"email": "david@tenant.example", "name": "David"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-unique-scope",
        correlationId="corr-unique-scope",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Capture final QASP edits",
        description="Capture final edits to the Quality Assurance Surveillance Plan.",
        mentionedPeople=["David"],
        dueText="before Friday",
        confidence=0.92,
        evidence=["explicit_task_language"],
    )
    wrong_search_result = _project(
        project_id="wrong-project",
        name="Unrelated Operations Support",
        search_score=0.78,
        search_rank=1,
        search_margin=0.08,
        people=[AssociatedPerson(name="David", email="david@tenant.example")],
        scopes=[
            ProjectScope(
                scopeId="wrong-scope",
                title="Operational Service Delivery",
                description="General service delivery.",
            )
        ],
    )
    expected = _project(
        project_id="expected-project",
        name="Facility Services",
        search_score=0.66,
        search_rank=9,
        search_margin=0.0,
        people=[AssociatedPerson(name="David", email="david@tenant.example")],
        scopes=[
            ProjectScope(
                scopeId="expected-scope",
                title="Quality Assurance Surveillance Plan",
                description="Continuous inspection and corrective action.",
            )
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [wrong_search_result, expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "expected-project"
    assert score.result.decision_reason == "explicit_unique_scope_title_reference"
    assert "explicit_unique_scope_title_reference" in score.result.evidence


async def test_explicit_project_name_overrides_incidental_generic_scope_heading():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="david@tenant.example",
        direction="received",
        graphEventId="graph-explicit-project-name",
        internetMessageId="<explicit-project-name@example.com>",
        messageId="explicit-project-name",
        subject="Gym Membership Services follow-up",
        bodyText=(
            "David, please confirm the operational service delivery analysis for "
            "Gym Membership Services before Friday."
        ),
        **{"from": {"email": "lead@external.example", "name": "External Lead"}},
        to=[{"email": "david@tenant.example", "name": "David"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-explicit-project-name",
        correlationId="corr-explicit-project-name",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Confirm the operational service delivery analysis",
        description="Confirm the operational service delivery analysis before Friday.",
        mentionedPeople=["David"],
        dueText="before Friday",
        confidence=0.92,
        evidence=["explicit_task_language"],
    )
    wrong_search_result = _project(
        project_id="wrong-project",
        name="Data Analysis Requirements",
        search_score=0.80,
        search_rank=1,
        search_margin=0.08,
        people=[AssociatedPerson(name="David", email="david@tenant.example")],
        scopes=[
            ProjectScope(
                scopeId="wrong-scope",
                title="Operational Service Delivery",
                description="General service delivery.",
            )
        ],
    )
    expected = _project(
        project_id="expected-project",
        name="Gym Membership Services",
        search_score=0.66,
        search_rank=11,
        search_margin=0.0,
        people=[AssociatedPerson(name="David", email="david@tenant.example")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [wrong_search_result, expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "expected-project"
    assert score.result.decision_reason == "explicit_project_name_reference"
    assert "explicit_project_name_reference" in score.result.evidence


async def test_generic_scope_heading_does_not_override_stronger_project_people_context():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="david@tenant.example",
        direction="received",
        graphEventId="graph-generic-scope",
        internetMessageId="<generic-scope@example.com>",
        messageId="generic-scope",
        subject="Security requirements analysis",
        bodyText=(
            "David, please reconcile the security requirements analysis with Priya "
            "before Friday."
        ),
        **{"from": {"email": "lead@external.example", "name": "External Lead"}},
        to=[{"email": "david@tenant.example", "name": "David"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-generic-scope",
        correlationId="corr-generic-scope",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Reconcile the security requirements analysis",
        description="Reconcile the security requirements analysis with Priya.",
        mentionedPeople=["David", "Priya"],
        dueText="before Friday",
        confidence=0.92,
        evidence=["explicit_task_language"],
    )
    wrong_search_result = _project(
        project_id="wrong-project",
        name="Internet Monitoring Service",
        search_score=0.78,
        search_rank=1,
        search_margin=0.08,
        people=[AssociatedPerson(name="Morgan", email="morgan@tenant.example")],
        scopes=[
            ProjectScope(
                scopeId="wrong-scope",
                title="Security Requirements",
                description="Personnel security requirements.",
            )
        ],
    )
    expected = _project(
        project_id="expected-project",
        name="Communications Capacity",
        search_score=0.70,
        search_rank=6,
        search_margin=0.0,
        people=[
            AssociatedPerson(name="David", email="david@tenant.example"),
            AssociatedPerson(name="Priya", email="priya@tenant.example"),
        ],
    )

    score = await score_project_candidates(
        request,
        [task],
        [wrong_search_result, expected],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.project.project_id == "expected-project"
    assert "explicit_unique_scope_title_reference" not in score.result.evidence


async def test_participant_and_weak_text_without_people_context_fail_closed():
    request = EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="david@tenant.example",
        direction="received",
        graphEventId="graph-no-project",
        internetMessageId="<no-project@example.com>",
        messageId="no-project",
        subject="Uncatalogued client request",
        bodyText="Please prepare the unrelated client package by Friday.",
        **{"from": {"email": "lead@external.example", "name": "External Lead"}},
        to=[{"email": "david@tenant.example", "name": ""}],
        cc=[],
        bcc=[],
        idempotencyKey="key-no-project",
        correlationId="corr-no-project",
    )
    task = ExtractedTaskCandidate(
        sourceTaskId="extracted-task-1",
        title="Prepare unrelated client package",
        description="Prepare the unrelated client package by Friday.",
        mentionedPeople=[],
        dueText="by Friday",
        confidence=0.92,
        evidence=["explicit_task_language"],
    )
    candidate = _project(
        project_id="nearest-project",
        name="General Client Support",
        search_score=0.71,
        search_rank=4,
        search_margin=0.0,
        people=[AssociatedPerson(name="David", email="david@tenant.example")],
    )

    score = await score_project_candidates(
        request,
        [task],
        [candidate],
        thread_context=None,
        threshold=0.80,
    )

    assert score is not None
    assert score.result.confidence < 0.80


def _project(
    *,
    project_id: str,
    name: str,
    search_score: float,
    search_rank: int,
    search_margin: float,
    people: list[AssociatedPerson],
    client_domains: list[str] | None = None,
    scopes: list[ProjectScope] | None = None,
) -> ProjectContext:
    return ProjectContext(
        projectId=project_id,
        projectName=name,
        description=f"{name} project work.",
        clientDomains=client_domains or [],
        associatedPeople=people,
        associatedManagers=[],
        scopes=scopes or [],
        searchScore=search_score,
        searchScoreRaw=search_score,
        searchRank=search_rank,
        searchMargin=search_margin,
    )
