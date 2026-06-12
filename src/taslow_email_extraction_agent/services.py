from __future__ import annotations

from dataclasses import dataclass

from taslow_email_extraction_agent.clients.project_client import ProjectClient
from taslow_email_extraction_agent.clients.project_search_client import ProjectSearchClient
from taslow_email_extraction_agent.clients.task_history_client import TaskHistoryClient
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.task_detection import TaskExtractor


@dataclass(slots=True)
class WorkflowServices:
    settings: Settings
    task_extractor: TaskExtractor
    project_client: ProjectClient
    task_history_client: TaskHistoryClient
    project_search_client: ProjectSearchClient | None = None
