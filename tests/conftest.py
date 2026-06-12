from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from taslow_email_extraction_agent.clients.project_client import InMemoryProjectClient
from taslow_email_extraction_agent.clients.task_history_client import EmptyTaskHistoryClient
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.task_detection import HeuristicTaskExtractor
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ProjectContext,
    ProjectScope,
)
from taslow_email_extraction_agent.services import WorkflowServices


@pytest.fixture
def project() -> ProjectContext:
    return ProjectContext(
        projectId="project-1",
        projectName="Cube Architecture Review",
        description="Electrical scope and architecture review for Cube location data.",
        associatedPeople=[
            AssociatedPerson(
                name="Tessa", email="tessa@tenant.com", aliases="Tess", role="Engineer"
            ),
            AssociatedPerson(name="Jesse", email="jesse@tenant.com", aliases="", role="Manager"),
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


@pytest.fixture
def services(project: ProjectContext) -> WorkflowServices:
    return WorkflowServices(
        settings=Settings(
            project_confidence_threshold=0.50,
            assignee_confidence_threshold=0.80,
        ),
        task_extractor=HeuristicTaskExtractor(),
        project_client=InMemoryProjectClient([project]),
        task_history_client=EmptyTaskHistoryClient(),
    )


@pytest.fixture
def base_request() -> EmailExtractionRequest:
    return EmailExtractionRequest(
        tenantId="tenant-1",
        mailbox="jesse@tenant.com",
        direction="sent",
        graphEventId="graph-1",
        internetMessageId="<msg-1@tenant.com>",
        messageId="msg-1",
        subject="Electrical scope update",
        bodyText="Tessa, please update the electrical scope by next Friday at 5.",
        sentDateTime=datetime(2026, 5, 15, 14, 30, tzinfo=ZoneInfo("America/New_York")),
        **{"from": {"email": "jesse@tenant.com", "name": "Jesse"}},
        to=[{"email": "tessa@tenant.com", "name": "Tessa"}],
        cc=[],
        bcc=[],
        idempotencyKey="key-1",
        correlationId="corr-1",
    )
