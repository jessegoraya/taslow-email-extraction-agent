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
    assert score.result.decision_reason == "participant_evidence_with_moderate_search"
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
    assert score.result.confidence == 0.81
    assert "project_selection_participant_tiebreak" in score.result.evidence


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
    assert "project_selection_weighted_evidence_tiebreak" in score.result.evidence


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


def _project(
    *,
    project_id: str,
    name: str,
    search_score: float,
    search_rank: int,
    search_margin: float,
    people: list[AssociatedPerson],
    client_domains: list[str] | None = None,
) -> ProjectContext:
    return ProjectContext(
        projectId=project_id,
        projectName=name,
        description=f"{name} project work.",
        clientDomains=client_domains or [],
        associatedPeople=people,
        associatedManagers=[],
        scopes=[],
        searchScore=search_score,
        searchScoreRaw=search_score,
        searchRank=search_rank,
        searchMargin=search_margin,
    )
