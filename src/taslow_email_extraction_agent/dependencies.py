from __future__ import annotations

from functools import lru_cache

from taslow_email_extraction_agent.clients.project_client import (
    HttpProjectClient,
    InMemoryProjectClient,
)
from taslow_email_extraction_agent.clients.project_search_client import AzureProjectSearchClient
from taslow_email_extraction_agent.clients.task_history_client import EmptyTaskHistoryClient
from taslow_email_extraction_agent.config import get_settings
from taslow_email_extraction_agent.executors.task_detection import FoundryTaskExtractor
from taslow_email_extraction_agent.services import WorkflowServices


@lru_cache
def build_services() -> WorkflowServices:
    settings = get_settings()
    if settings.project_service_base_url:
        project_client = HttpProjectClient(
            settings.project_service_base_url,
            api_key=settings.taslow_service_api_key,
        )
    else:
        project_client = InMemoryProjectClient()

    project_search_client = None
    if settings.project_search_provider in {"azure-ai-search", "shadow"}:
        project_search_client = AzureProjectSearchClient(settings)

    return WorkflowServices(
        settings=settings,
        task_extractor=FoundryTaskExtractor(settings),
        project_client=project_client,
        task_history_client=EmptyTaskHistoryClient(),
        project_search_client=project_search_client,
    )


def get_workflow_services() -> WorkflowServices:
    return build_services()
