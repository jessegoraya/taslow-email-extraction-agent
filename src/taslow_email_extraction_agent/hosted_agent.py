from __future__ import annotations

import logging
from typing import Any

from taslow_email_extraction_agent.dependencies import get_workflow_services
from taslow_email_extraction_agent.models import EmailExtractionRequest
from taslow_email_extraction_agent.services import WorkflowServices
from taslow_email_extraction_agent.workflow import run_email_extraction

logger = logging.getLogger(__name__)


async def process_email_invocation(
    payload: object,
    *,
    services: WorkflowServices | None = None,
    correlation_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the existing email workflow through the Foundry Invocations contract."""

    if not isinstance(payload, dict):
        return 400, _error(
            "INVALID_REQUEST",
            "The invocation body must be an email extraction object.",
            correlation_id,
        )

    try:
        request = EmailExtractionRequest.model_validate(payload)
    except ValueError:
        return 400, _error(
            "INVALID_REQUEST",
            "The email extraction payload is invalid.",
            correlation_id,
        )

    effective_correlation_id = correlation_id or request.correlation_id
    try:
        response = await run_email_extraction(request, services or get_workflow_services())
    except Exception:
        logger.exception(
            "Foundry email extraction invocation failed. correlation_id=%s",
            effective_correlation_id,
        )
        return 500, _error(
            "AGENT_EXECUTION_FAILED",
            "The email extraction agent could not process this request.",
            effective_correlation_id,
        )

    return 200, response.to_jsonable()


def _error(code: str, message: str, correlation_id: str | None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "correlationId": correlation_id,
        }
    }
