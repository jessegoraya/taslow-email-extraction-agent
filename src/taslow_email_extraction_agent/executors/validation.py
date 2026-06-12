from __future__ import annotations

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.models import ExtractedTaskAssignment


@step(name="ResultValidationExecutor")
async def validate_assignments(
    assignments: list[ExtractedTaskAssignment],
    settings: Settings,
) -> list[ExtractedTaskAssignment]:
    deduped: dict[tuple[str, str, str], ExtractedTaskAssignment] = {}
    for assignment in assignments:
        if assignment.assignee_confidence < settings.assignee_confidence_threshold:
            continue
        if assignment.needs_review:
            continue
        key = (
            assignment.source_task_id,
            assignment.assignee_email.lower(),
            assignment.project_id,
        )
        existing = deduped.get(key)
        if existing is None or assignment.overall_confidence > existing.overall_confidence:
            deduped[key] = assignment
    return list(deduped.values())
