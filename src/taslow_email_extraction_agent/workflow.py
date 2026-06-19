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
    EmailExtractionRequest,
    EmailExtractionResponse,
    ExtractedTaskAssignment,
    ExtractionDiagnostics,
    ExtractionStatus,
    ProjectMatchResult,
    ProjectScope,
    ProjectScoringDiagnostics,
    ThreadContext,
)
from taslow_email_extraction_agent.services import WorkflowServices


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
    valid_assignments = await validate_assignments(assignments, services.settings)
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

    for task in tasks:
        scored_project = project
        try:
            scored_project = await _apply_scope_search_scores(
                request, services, project, task.description
            )
        except ProjectSearchUnavailable:
            if services.project_search_client:
                raise

        scope, scope_confidence, scope_evidence = await match_scope_area(
            task, scored_project, thread_context
        )
        assignees = await resolve_assignees(request, task, scored_project)
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

    return assignments


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
