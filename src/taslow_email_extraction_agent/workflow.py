from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from taslow_email_extraction_agent.agent_framework_compat import workflow
from taslow_email_extraction_agent.clients.project_search_client import ProjectSearchUnavailable
from taslow_email_extraction_agent.executors.assignee_resolution import resolve_assignees
from taslow_email_extraction_agent.executors.due_date import normalize_due_date
from taslow_email_extraction_agent.executors.project_matching import (
    ProjectScore,
    retrieve_project_candidates,
    score_project_candidates,
)
from taslow_email_extraction_agent.executors.scope_matching import match_scope_area
from taslow_email_extraction_agent.executors.task_detection import detect_tasks
from taslow_email_extraction_agent.executors.validation import validate_assignments
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    EmailExtractionResponse,
    ExtractedTaskAssignment,
    ExtractedTaskCandidate,
    ExtractionDiagnostics,
    ExtractionStatus,
    ProjectContext,
    ProjectMatchResult,
    ProjectScope,
    ProjectScoringDiagnostics,
    ThreadContext,
)
from taslow_email_extraction_agent.services import WorkflowServices
from taslow_email_extraction_agent.text_utils import (
    split_newest_and_quoted_text,
    task_context_text,
)


@dataclass(slots=True)
class WorkflowInput:
    request: EmailExtractionRequest
    services: WorkflowServices


async def run_email_extraction(
    request: EmailExtractionRequest,
    services: WorkflowServices,
) -> EmailExtractionResponse:
    message = WorkflowInput(request=request, services=services)
    if hasattr(email_extraction_workflow, "run"):
        result = await email_extraction_workflow.run(message)
        outputs = result.get_outputs()
        if outputs:
            return outputs[0]
        raise ValueError("Agent Framework workflow completed without an output.")
    return await email_extraction_workflow(message)


@workflow(
    name="TaslowEmailExtractionWorkflow", description="Extract Taslow project tasks from email."
)
async def email_extraction_workflow(message: WorkflowInput) -> EmailExtractionResponse:
    request = message.request
    services = message.services
    warnings: list[str] = []

    tasks = await detect_tasks(request, services.task_extractor)
    task_extractor_info = getattr(services.task_extractor, "last_run_info", None)
    if task_extractor_info and task_extractor_info.warning:
        warnings.append(task_extractor_info.warning)
    if not tasks:
        return _response(
            request=request,
            services=services,
            status=ExtractionStatus.NO_TASK_FOUND,
            task_candidate_count=0,
            project_match=None,
            assignments=[],
            stopped_after="TaskDetectionExecutor",
            warnings=warnings,
        )

    if not request.visible_recipients:
        warnings.append("participant_context_missing")

    thread_context = await services.task_history_client.get_thread_context(request)
    search_text = " ".join([request.combined_text, *[task.description for task in tasks]])
    try:
        projects = await _retrieve_projects(request, services, search_text)
    except ProjectSearchUnavailable:
        warnings.append("azure_ai_search_dependency_failure")
        return _retryable_response(
            request=request,
            services=services,
            task_candidate_count=len(tasks),
            warnings=warnings,
            stopped_after="ProjectCandidateRetrievalExecutor",
        )

    candidates = await retrieve_project_candidates(projects)
    project_score = await score_project_candidates(
        request,
        tasks,
        candidates,
        thread_context,
        services.settings.project_confidence_threshold,
    )

    if (
        not project_score
        or project_score.result.confidence < services.settings.project_confidence_threshold
    ):
        return _response(
            request=request,
            services=services,
            status=ExtractionStatus.NO_PROJECT_MATCH,
            task_candidate_count=len(tasks),
            project_match=project_score.result if project_score else None,
            project_candidates=project_score.candidate_results if project_score else [],
            assignments=[],
            stopped_after="ProjectScoringExecutor",
            warnings=warnings,
        )

    try:
        assignments = await _build_assignments(request, services, tasks, project_score)
    except ProjectSearchUnavailable:
        warnings.append("azure_ai_search_dependency_failure")
        return _retryable_response(
            request=request,
            services=services,
            task_candidate_count=len(tasks),
            warnings=warnings,
            stopped_after="ScopeAreaMatchingExecutor",
        )
    valid_assignments = await validate_assignments(assignments, services.settings, request)
    status = (
        ExtractionStatus.TASKS_READY if valid_assignments else ExtractionStatus.NO_PROJECT_MATCH
    )
    stopped_after = None if valid_assignments else "ResultValidationExecutor"
    if status == ExtractionStatus.TASKS_READY:
        _record_thread_context(request, services, project_score, valid_assignments)

    return _response(
        request=request,
        services=services,
        status=status,
        task_candidate_count=len(tasks),
        project_match=project_score.result,
        project_candidates=project_score.candidate_results,
        assignments=valid_assignments,
        stopped_after=stopped_after,
        warnings=warnings,
    )


async def _build_assignments(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    tasks: list,
    project_score: ProjectScore,
) -> list[ExtractedTaskAssignment]:
    project = project_score.project
    assignments: list[ExtractedTaskAssignment] = []
    thread_context = await services.task_history_client.get_thread_context(request)
    used_multi_task_assignees: set[str] = set()

    for task_index, task in enumerate(tasks):
        scored_project = project
        try:
            scored_project = await _apply_scope_search_scores(
                request, services, project, task_context_text(request.body_text, task.description)
            )
        except ProjectSearchUnavailable:
            if services.project_search_client:
                raise

        scope = _match_explicit_scope_reference(request, scored_project)
        if scope:
            scope_confidence = 0.98
            scope_evidence = ["explicit_scope_title_reference"]
            if scope.search_score:
                scope_evidence.append("azure_ai_search_scope_similarity")
            if scope.search_rank == 1 and scope.search_score:
                scope_evidence.append("top_scope_search_candidate")
        else:
            scope, scope_confidence, scope_evidence = await match_scope_area(
                task, scored_project, thread_context
            )
        if services.scope_reranker:
            scope, scope_confidence, scope_evidence = await services.scope_reranker.rerank_scope(
                request,
                task,
                scored_project,
                scope,
                scope_confidence,
                scope_evidence,
            )
        assignees = await resolve_assignees(request, task, scored_project)
        if services.assignee_reranker:
            assignees = await services.assignee_reranker.rerank_assignees(
                request,
                task,
                scored_project,
                assignees,
            )
        assignees = _apply_ordered_multi_task_recipient(
            request,
            scored_project,
            task,
            assignees,
            task_index,
            len(tasks),
        )
        assignees = _normalize_assignees_for_task(
            task, assignees, len(tasks), used_multi_task_assignees
        )
        if len(tasks) > 1 and len(assignees) == 1:
            used_multi_task_assignees.add(assignees[0][0].email)
        due_date, due_confidence, due_evidence = await normalize_due_date(request, task)

        for person, assignee_confidence, assignee_evidence in assignees:
            overall = _overall_confidence(
                task_confidence=task.confidence,
                project_confidence=project_score.result.confidence,
                scope_confidence=scope_confidence,
                assignee_confidence=assignee_confidence,
                due_confidence=due_confidence,
            )
            assignments.append(
                ExtractedTaskAssignment(
                    sourceTaskId=task.source_task_id,
                    title=task.title,
                    description=task.description,
                    projectId=scored_project.project_id,
                    scopeId=scope.scope_id
                    if scope and scope_confidence >= services.settings.scope_confidence_threshold
                    else None,
                    scopeConfidence=scope_confidence if scope else None,
                    assigneeEmail=person.email,
                    assigneeName=person.name,
                    assigneeConfidence=assignee_confidence,
                    dueDate=due_date,
                    dueDateConfidence=due_confidence,
                    overallConfidence=overall,
                    evidence=[
                        *task.evidence,
                        *project_score.result.evidence,
                        *scope_evidence,
                        *assignee_evidence,
                        *due_evidence,
                    ],
                    needsReview=assignee_confidence
                    < services.settings.assignee_confidence_threshold,
                )
            )

    return _rewrite_ordered_multi_task_assignments(request, project, assignments, len(tasks))


def _apply_ordered_multi_task_recipient(
    request: EmailExtractionRequest,
    project,
    task: ExtractedTaskCandidate,
    assignees: list[tuple[AssociatedPerson, float, list[str]]],
    task_index: int,
    task_count: int,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if task_count <= 1 or task_index >= len(request.to):
        return assignees
    if len(request.to) < task_count:
        return assignees

    task_text = " ".join([task.title, task.description]).lower()
    if any("sender_self_ownership_signal" in evidence for _p, _s, evidence in assignees):
        return assignees

    project_people_by_email = {person.email: person for person in project.people if person.email}
    if any(
        _task_text_mentions_person(task_text, person) for person in project_people_by_email.values()
    ):
        return assignees

    recipient = request.to[task_index]
    person = project_people_by_email.get(recipient.email)
    if not person:
        return assignees

    existing = next(
        (assignee for assignee in assignees if assignee[0].email == person.email),
        None,
    )
    if existing:
        _person, confidence, evidence = existing
        return [
            (
                person,
                max(confidence, 0.89),
                sorted(
                    {
                        *evidence,
                        "ordered_multi_task_recipient_assignment",
                    }
                ),
            )
        ]
    return [
        (
            person,
            0.89,
            ["ordered_multi_task_recipient_assignment"],
        )
    ]


def _rewrite_ordered_multi_task_assignments(
    request: EmailExtractionRequest,
    project,
    assignments: list[ExtractedTaskAssignment],
    task_count: int,
) -> list[ExtractedTaskAssignment]:
    if task_count <= 1 or len(assignments) <= 1:
        return assignments
    if len(request.to) < len(assignments):
        return assignments

    project_people_by_email = {person.email: person for person in project.people if person.email}
    ordered_people = [
        project_people_by_email.get(recipient.email) for recipient in request.to[: len(assignments)]
    ]
    if any(person is None for person in ordered_people):
        return assignments

    rewritten: list[ExtractedTaskAssignment] = []
    for index, assignment in enumerate(assignments):
        if _assignment_has_strong_assignee_evidence(assignment):
            rewritten.append(assignment)
            continue
        person = ordered_people[index]
        assert person is not None
        rewritten.append(
            assignment.model_copy(
                update={
                    "assignee_email": person.email,
                    "assignee_name": person.name,
                    "assignee_confidence": max(assignment.assignee_confidence, 0.89),
                    "evidence": sorted(
                        {
                            *assignment.evidence,
                            "ordered_multi_task_recipient_assignment",
                            "multi_task_assignee_normalized",
                        }
                    ),
                    "needs_review": False,
                }
            )
        )
    return rewritten


def _assignment_has_strong_assignee_evidence(assignment: ExtractedTaskAssignment) -> bool:
    strong_evidence = {
        "direct_address_assignment",
        "delegated_assignment_language",
        "need_named_person_to_act",
        "requested_actor_assignment_language",
        "named_owner_drive_assignment",
        "modal_named_actor_assignment",
        "same_block_generic_request_at_mention",
        "subject_matter_owner_signal",
        "beneficiary_or_owner_signal",
        "named_request_actor_signal",
        "sender_self_ownership_signal",
    }
    return bool(strong_evidence & set(assignment.evidence))


def _normalize_assignees_for_task(
    task: ExtractedTaskCandidate,
    assignees: list[tuple[AssociatedPerson, float, list[str]]],
    task_count: int,
    used_assignee_emails: set[str] | None = None,
) -> list[tuple[AssociatedPerson, float, list[str]]]:
    if task_count <= 1 or len(assignees) <= 1:
        return assignees

    used_assignee_emails = used_assignee_emails or set()
    task_text = " ".join([task.title, task.description, *task.mentioned_people]).lower()
    strongly_tied = [
        assignee
        for assignee in assignees
        if _assignee_has_task_specific_evidence(task_text, assignee)
    ]
    unused = [assignee for assignee in assignees if assignee[0].email not in used_assignee_emails]
    selected = strongly_tied or unused or assignees[:1]
    return [
        (
            person,
            confidence,
            sorted({*evidence, "multi_task_assignee_normalized"}),
        )
        for person, confidence, evidence in selected[:1]
    ]


def _assignee_has_task_specific_evidence(
    task_text: str,
    assignee: tuple[AssociatedPerson, float, list[str]],
) -> bool:
    person, confidence, evidence = assignee
    evidence_set = set(evidence)
    strong_evidence = {
        "direct_address_assignment",
        "delegated_assignment_language",
        "need_named_person_to_act",
        "requested_actor_assignment_language",
        "named_owner_drive_assignment",
        "modal_named_actor_assignment",
        "named_person_action_language",
        "same_block_generic_request_at_mention",
        "subject_matter_owner_signal",
        "beneficiary_or_owner_signal",
        "named_request_actor_signal",
    }
    if evidence_set & strong_evidence and confidence >= 0.88:
        return True

    return _task_text_mentions_person(task_text, person)


def _task_text_mentions_person(task_text: str, person: AssociatedPerson) -> bool:
    person_refs = [person.name.lower(), person.email.lower()]
    if person.aliases:
        person_refs.extend(alias.strip().lower() for alias in person.aliases.split(","))
    if any(ref and ref in task_text for ref in person_refs):
        return True
    if person.name:
        first_name = person.name.split()[0].lower()
        if len(first_name) > 2 and first_name in task_text:
            return True
    return False


async def _retrieve_projects(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    search_text: str,
):
    if not services.project_search_client:
        return await services.project_client.get_active_projects(request.tenant_id)

    candidates = await services.project_search_client.search_projects(
        request.tenant_id, search_text
    )
    projects_by_id = {
        project.project_id: project
        for project in await services.project_client.get_project_context_batch(
            request.tenant_id, [candidate.project_id for candidate in candidates]
        )
    }
    projects = []
    for candidate in candidates:
        project = projects_by_id.get(candidate.project_id)
        if project:
            projects.append(
                project.model_copy(
                    update={
                        "search_score": candidate.score,
                        "search_score_raw": candidate.score_raw,
                        "search_rank": candidate.rank,
                        "search_margin": candidate.score_margin,
                    }
                )
            )
    return projects


async def _apply_scope_search_scores(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    project,
    task_text: str,
):
    if not services.project_search_client:
        return project

    candidates = await services.project_search_client.search_scopes(
        request.tenant_id, project.project_id, task_text
    )
    scores_by_scope = {candidate.scope_id: candidate for candidate in candidates}
    scopes: list[ProjectScope] = [
        scope.model_copy(
            update={
                "search_score": scores_by_scope[scope.scope_id].score,
                "search_score_raw": scores_by_scope[scope.scope_id].score_raw,
                "search_rank": scores_by_scope[scope.scope_id].rank,
                "search_margin": scores_by_scope[scope.scope_id].score_margin,
            }
        )
        if scope.scope_id in scores_by_scope
        else scope
        for scope in project.scopes
    ]
    return project.model_copy(update={"scopes": scopes})


def _match_explicit_scope_reference(
    request: EmailExtractionRequest,
    project: ProjectContext,
) -> ProjectScope | None:
    newest_body, _quoted_context = split_newest_and_quoted_text(request.body_text)
    authored_text = _normalize_scope_reference_text(" ".join([request.subject, newest_body]))
    if not authored_text:
        return None

    matches = [
        scope
        for scope in project.scopes
        if _normalize_scope_reference_text(scope.title)
        and _contains_normalized_phrase(
            authored_text,
            _normalize_scope_reference_text(scope.title),
        )
    ]
    if not matches:
        return None

    longest_length = max(len(_normalize_scope_reference_text(scope.title)) for scope in matches)
    longest_matches = [
        scope
        for scope in matches
        if len(_normalize_scope_reference_text(scope.title)) == longest_length
    ]
    return longest_matches[0] if len(longest_matches) == 1 else None


def _normalize_scope_reference_text(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value.lower()).split()
    )


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _overall_confidence(
    task_confidence: float,
    project_confidence: float,
    scope_confidence: float,
    assignee_confidence: float,
    due_confidence: float | None,
) -> float:
    due = due_confidence if due_confidence is not None else 0.70
    scope = scope_confidence if scope_confidence else 0.75
    return round(
        min(
            1.0,
            (task_confidence * 0.20)
            + (project_confidence * 0.35)
            + (scope * 0.10)
            + (assignee_confidence * 0.25)
            + (due * 0.10),
        ),
        3,
    )


def _response(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    status: ExtractionStatus,
    task_candidate_count: int,
    project_match: ProjectMatchResult | None,
    assignments: list[ExtractedTaskAssignment],
    stopped_after: str | None,
    warnings: list[str],
    project_candidates: list[ProjectMatchResult] | None = None,
) -> EmailExtractionResponse:
    task_info = getattr(services.task_extractor, "last_run_info", None)
    return EmailExtractionResponse(
        agentRunId=str(uuid4()),
        status=status,
        tenantId=request.tenant_id,
        graphEventId=request.graph_event_id,
        internetMessageId=request.internet_message_id,
        messageId=request.message_id,
        taskCandidateCount=task_candidate_count,
        projectMatch=project_match,
        tasks=assignments,
        diagnostics=ExtractionDiagnostics(
            model=services.settings.azure_ai_model_deployment_name,
            taskExtractorProvider=task_info.provider if task_info else None,
            modelDeployment=task_info.model_deployment if task_info else None,
            modelFallbackUsed=task_info.fallback_used if task_info else False,
            modelInputTokenCount=task_info.input_tokens if task_info else None,
            modelOutputTokenCount=task_info.output_tokens if task_info else None,
            taskDetectionRecoveryAttempted=task_info.recovery_attempted
            if task_info
            else False,
            taskDetectionRecoverySucceeded=task_info.recovery_succeeded
            if task_info
            else False,
            taskDetectionRecoveryReason=task_info.recovery_reason if task_info else None,
            projectThreshold=services.settings.project_confidence_threshold,
            scopeThreshold=services.settings.scope_confidence_threshold,
            assigneeThreshold=services.settings.assignee_confidence_threshold,
            dueDateThreshold=services.settings.due_date_confidence_threshold,
            projectHydrationProvider="project-agent-context"
            if services.project_search_client
            else "active-projects",
            searchQueryCount=1 if services.project_search_client and task_candidate_count else 0,
            scopeSearchQueryCount=len(assignments)
            if services.project_search_client and assignments
            else 0,
            projectScoring=_project_scoring_diagnostics(project_match),
            projectCandidateScores=project_candidates or [],
            stoppedAfter=stopped_after,
            warnings=warnings,
        ),
    )


def _retryable_response(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    task_candidate_count: int,
    warnings: list[str],
    stopped_after: str,
) -> EmailExtractionResponse:
    task_info = getattr(services.task_extractor, "last_run_info", None)
    return EmailExtractionResponse(
        agentRunId=str(uuid4()),
        status=ExtractionStatus.RETRYABLE,
        tenantId=request.tenant_id,
        graphEventId=request.graph_event_id,
        internetMessageId=request.internet_message_id,
        messageId=request.message_id,
        taskCandidateCount=task_candidate_count,
        projectMatch=None,
        tasks=[],
        diagnostics=ExtractionDiagnostics(
            model=services.settings.azure_ai_model_deployment_name,
            taskExtractorProvider=task_info.provider if task_info else None,
            modelDeployment=task_info.model_deployment if task_info else None,
            modelFallbackUsed=task_info.fallback_used if task_info else False,
            modelInputTokenCount=task_info.input_tokens if task_info else None,
            modelOutputTokenCount=task_info.output_tokens if task_info else None,
            taskDetectionRecoveryAttempted=task_info.recovery_attempted
            if task_info
            else False,
            taskDetectionRecoverySucceeded=task_info.recovery_succeeded
            if task_info
            else False,
            taskDetectionRecoveryReason=task_info.recovery_reason if task_info else None,
            projectThreshold=services.settings.project_confidence_threshold,
            scopeThreshold=services.settings.scope_confidence_threshold,
            assigneeThreshold=services.settings.assignee_confidence_threshold,
            dueDateThreshold=services.settings.due_date_confidence_threshold,
            projectHydrationProvider="project-agent-context"
            if services.project_search_client
            else "active-projects",
            searchQueryCount=1 if services.project_search_client and task_candidate_count else 0,
            scopeSearchQueryCount=0,
            stoppedAfter=stopped_after,
            warnings=warnings,
            retrySchedule=["PT10M", "PT4H", "PT24H"]
            if services.settings.agent_search_dependency_retry_enabled
            else [],
            manualExecutionRequired=False,
        ),
    )


def _project_scoring_diagnostics(
    project_match: ProjectMatchResult | None,
) -> ProjectScoringDiagnostics | None:
    if not project_match:
        return None
    return ProjectScoringDiagnostics(
        searchScoreRaw=project_match.search_score_raw,
        searchScoreNormalized=project_match.search_score_normalized,
        searchRank=project_match.search_rank,
        searchMargin=project_match.search_margin,
        participantScore=project_match.participant_score,
        peopleContextScore=project_match.people_context_score,
        clientDomainScore=project_match.client_domain_score,
        lexicalScore=project_match.lexical_score,
        threshold=project_match.threshold,
        decisionReason=project_match.decision_reason,
    )


def _record_thread_context(
    request: EmailExtractionRequest,
    services: WorkflowServices,
    project_score: ProjectScore,
    assignments: list[ExtractedTaskAssignment],
) -> None:
    recorder = getattr(services.task_history_client, "record_thread_context", None)
    if not recorder:
        return
    scope_id = next(
        (assignment.scope_id for assignment in assignments if assignment.scope_id),
        None,
    )
    recorder(
        request,
        ThreadContext(
            projectId=project_score.project.project_id,
            scopeId=scope_id,
            confidence=project_score.result.confidence,
        ),
    )
