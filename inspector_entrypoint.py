"""Agent Inspector entrypoint for the Taslow email extraction workflow.

This file is intentionally separate from the production FastAPI app. Agent Inspector launches
an Agent Framework entity through `agentdev`, sends chat text to it, and visualizes the
workflow events. The production endpoint still lives in `taslow_email_extraction_agent.app`.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from agent_framework import FunctionalWorkflowAgent, workflow
from agent_framework_foundry_hosting import ResponsesHostServer

from taslow_email_extraction_agent.clients.project_client import InMemoryProjectClient
from taslow_email_extraction_agent.clients.task_history_client import EmptyTaskHistoryClient
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.assignee_resolution import resolve_assignees
from taslow_email_extraction_agent.executors.due_date import normalize_due_date
from taslow_email_extraction_agent.executors.project_matching import (
    retrieve_project_candidates,
    score_project_candidates,
)
from taslow_email_extraction_agent.executors.scope_matching import match_scope_area
from taslow_email_extraction_agent.executors.task_detection import (
    HeuristicTaskExtractor,
    detect_tasks,
)
from taslow_email_extraction_agent.executors.validation import validate_assignments
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ExtractedTaskAssignment,
    ExtractionStatus,
    ProjectContext,
    ProjectScope,
)
from taslow_email_extraction_agent.services import WorkflowServices


@workflow(
    name="TaslowEmailExtractionInspectorWorkflow",
    description="Visual debug workflow for Taslow email-to-task extraction.",
)
async def taslow_email_extraction_inspector_workflow(messages=None) -> str:
    request = _build_request_from_inspector_message(messages)
    services = _build_sample_services()

    tasks = await detect_tasks(request, services.task_extractor)
    if not tasks:
        return _summary(ExtractionStatus.NO_TASK_FOUND.value, 0, 0)

    thread_context = await services.task_history_client.get_thread_context(request)
    projects = await services.project_client.get_active_projects(request.tenant_id)
    candidates = await retrieve_project_candidates(projects)
    project_score = await score_project_candidates(request, tasks, candidates, thread_context)
    if (
        not project_score
        or project_score.result.confidence < services.settings.project_confidence_threshold
    ):
        return _summary(ExtractionStatus.NO_PROJECT_MATCH.value, len(tasks), 0)

    assignments: list[ExtractedTaskAssignment] = []
    for task in tasks:
        scope, scope_confidence, scope_evidence = await match_scope_area(
            task,
            project_score.project,
            thread_context,
        )
        assignees = await resolve_assignees(request, task, project_score.project)
        due_date, due_confidence, due_evidence = await normalize_due_date(request, task)

        for person, assignee_confidence, assignee_evidence in assignees:
            assignments.append(
                ExtractedTaskAssignment(
                    sourceTaskId=task.source_task_id,
                    title=task.title,
                    description=task.description,
                    projectId=project_score.project.project_id,
                    scopeId=scope.scope_id if scope and scope_confidence >= 0.75 else None,
                    scopeConfidence=scope_confidence if scope else None,
                    assigneeEmail=person.email,
                    assigneeName=person.name,
                    assigneeConfidence=assignee_confidence,
                    dueDate=due_date,
                    dueDateConfidence=due_confidence,
                    overallConfidence=min(
                        1.0,
                        round(
                            (task.confidence * 0.20)
                            + (project_score.result.confidence * 0.35)
                            + ((scope_confidence or 0.75) * 0.10)
                            + (assignee_confidence * 0.25)
                            + ((due_confidence or 0.70) * 0.10),
                            3,
                        ),
                    ),
                    evidence=[
                        *task.evidence,
                        *project_score.result.evidence,
                        *scope_evidence,
                        *assignee_evidence,
                        *due_evidence,
                    ],
                )
            )

    valid_assignments = await validate_assignments(assignments, services.settings)
    status = (
        ExtractionStatus.TASKS_READY.value
        if valid_assignments
        else ExtractionStatus.NO_PROJECT_MATCH.value
    )
    return _summary(status, len(tasks), len(valid_assignments))


agent = FunctionalWorkflowAgent(
    taslow_email_extraction_inspector_workflow,
    name="Taslow Email Extraction Agent",
    description="Visual Agent Inspector wrapper around the Taslow extraction workflow.",
    context_providers=[],
)


def _build_request_from_inspector_message(messages) -> EmailExtractionRequest:
    body_text = _extract_text(messages) or (
        "Tessa, please update the electrical scope by next Friday at 5."
    )
    return EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="sent",
        graphEventId="inspector-graph-event",
        internetMessageId="<inspector-message@taslow.local>",
        messageId="inspector-message",
        subject="Inspector email extraction sample",
        bodyText=body_text,
        sentDateTime=datetime(2026, 5, 15, 14, 30, tzinfo=ZoneInfo("America/New_York")),
        **{"from": {"email": "jesse@tenant.com", "name": "Jesse"}},
        to=[{"email": "tessa@tenant.com", "name": "Tessa"}],
        cc=[],
        bcc=[],
        idempotencyKey="inspector-key",
        correlationId="inspector-correlation",
    )


def _build_sample_services() -> WorkflowServices:
    project = ProjectContext(
        projectId="project-1",
        projectName="Cube Architecture Review",
        description="Electrical scope and architecture review for Cube location data.",
        associatedPeople=[
            AssociatedPerson(name="Tessa", email="tessa@tenant.com", aliases="Tess"),
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", aliases=""),
        ],
        associatedManagers=[],
        scopes=[
            ProjectScope(
                scopeId="scope-1",
                title="Electrical Scope",
                description="Electrical scope updates and review.",
                groupTaskSetId="gts-1",
            )
        ],
    )
    return WorkflowServices(
        settings=Settings(
            project_confidence_threshold=0.50,
            assignee_confidence_threshold=0.80,
        ),
        task_extractor=HeuristicTaskExtractor(),
        project_client=InMemoryProjectClient([project]),
        task_history_client=EmptyTaskHistoryClient(),
    )


def _extract_text(messages) -> str:
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list) and messages:
        return _extract_text(messages[-1])
    if isinstance(messages, dict):
        content = messages.get("content") or messages.get("text") or ""
        if isinstance(content, list):
            return " ".join(_extract_text(item) for item in content)
        return str(content)
    text = getattr(messages, "text", None)
    if text:
        return str(text)
    content = getattr(messages, "content", None)
    if content:
        return _extract_text(content)
    return str(messages)


def _summary(status: str, task_count: int, assignment_count: int) -> str:
    return (
        f"Taslow extraction status: {status}. "
        f"Task candidates: {task_count}. "
        f"Assignments ready: {assignment_count}."
    )


if __name__ == "__main__":
    server = ResponsesHostServer(agent)
    server.run()
