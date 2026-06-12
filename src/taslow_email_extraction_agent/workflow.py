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
    project_score = await score_project_candidates(request, tasks, candidates, thread_context)

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
    projects = []
    for candidate in candidates:
        project = await services.project_client.get_project_detail(
            request.tenant_id, candidate.project_id
        )
        if project:
            projects.append(project.model_copy(update={"search_score": candidate.score}))
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
    scores_by_scope = {candidate.scope_id: candidate.score for candidate in candidates}
    scopes: list[ProjectScope] = [
        scope.model_copy(update={"search_score": scores_by_scope.get(scope.scope_id)})
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
            projectThreshold=services.settings.project_confidence_threshold,
            scopeThreshold=services.settings.scope_confidence_threshold,
            assigneeThreshold=services.settings.assignee_confidence_threshold,
            dueDateThreshold=services.settings.due_date_confidence_threshold,
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
            projectThreshold=services.settings.project_confidence_threshold,
            scopeThreshold=services.settings.scope_confidence_threshold,
            assigneeThreshold=services.settings.assignee_confidence_threshold,
            dueDateThreshold=services.settings.due_date_confidence_threshold,
            stoppedAfter=stopped_after,
            warnings=warnings,
            retrySchedule=["PT10M", "PT4H", "PT24H"]
            if services.settings.agent_search_dependency_retry_enabled
            else [],
            manualExecutionRequired=False,
        ),
    )
