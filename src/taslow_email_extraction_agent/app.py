from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException

from taslow_email_extraction_agent.dependencies import get_workflow_services
from taslow_email_extraction_agent.models import EmailExtractionRequest, EmailExtractionResponse
from taslow_email_extraction_agent.services import WorkflowServices
from taslow_email_extraction_agent.workflow import run_email_extraction

app = FastAPI(
    title="Taslow Email Extraction Agent",
    version="0.1.0",
    description="Microsoft Agent Framework workflow service for Taslow email-to-task extraction.",
)

WORKFLOW_SERVICES_DEPENDENCY = Depends(get_workflow_services)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/email-extractions",
    response_model=EmailExtractionResponse,
    response_model_by_alias=True,
)
async def create_email_extraction(
    request: EmailExtractionRequest,
    x_correlation_id: str | None = Header(default=None),
    services: WorkflowServices = WORKFLOW_SERVICES_DEPENDENCY,
) -> EmailExtractionResponse:
    if x_correlation_id and not request.correlation_id:
        request.correlation_id = x_correlation_id
    try:
        return await run_email_extraction(request, services)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
