from __future__ import annotations

import json
from time import time

import httpx
from azure.core.credentials import AccessToken

from taslow_email_extraction_agent.clients.project_client import HttpProjectClient


class FakeCredential:
    def __init__(self) -> None:
        self.scopes: list[tuple[str, ...]] = []

    async def get_token(self, *scopes: str) -> AccessToken:
        self.scopes.append(scopes)
        return AccessToken("project-token", int(time()) + 3600)


async def test_project_context_uses_managed_identity_and_maps_response() -> None:
    credential = FakeCredential()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer project-token"
        assert request.url.path.endswith("/internal/projects/agent-context/batch")
        payload = json.loads(request.content)
        assert payload["tenantId"] == "tenant-1"
        return httpx.Response(
            200,
            json={
                "projects": [
                    {
                        "projectId": "project-1",
                        "projectName": "BloomSky migration",
                        "scopes": [
                            {
                                "scopeId": "scope-1",
                                "scopeTitle": "Readiness",
                                "groupTaskSetId": "gts-1",
                            }
                        ],
                    }
                ]
            },
        )

    client = HttpProjectClient(
        "https://apim.example.test/FunctionProjectApp",
        token_scope="https://management.azure.com/.default",
        credential=credential,
        transport=httpx.MockTransport(handler),
    )

    projects = await client.get_project_context_batch("tenant-1", ["project-1"])

    assert credential.scopes == [("https://management.azure.com/.default",)]
    assert projects[0].project_id == "project-1"
    assert projects[0].scopes[0].group_task_set_id == "gts-1"


async def test_participant_candidates_use_managed_identity_and_normalize_request() -> None:
    credential = FakeCredential()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer project-token"
        assert request.url.path.endswith("/internal/projects/participant-candidates")
        payload = json.loads(request.content)
        assert payload == {
            "tenantId": "tenant-1",
            "participantEmails": [
                "alex@bloomsky.onmicrosoft.com",
                "paco@bloomsky.onmicrosoft.com",
            ],
        }
        return httpx.Response(
            200,
            json={
                "projects": [
                    {
                        "projectId": "va-radiology",
                        "matchedParticipantEmails": payload["participantEmails"],
                    }
                ]
            },
        )

    client = HttpProjectClient(
        "https://apim.example.test/FunctionProjectApp",
        token_scope="https://management.azure.com/.default",
        credential=credential,
        transport=httpx.MockTransport(handler),
    )

    project_ids = await client.get_participant_project_candidates(
        "tenant-1",
        [
            "PACO@bloomsky.onmicrosoft.com",
            "alex@bloomsky.onmicrosoft.com",
            "paco@bloomsky.onmicrosoft.com",
        ],
    )

    assert project_ids == ["va-radiology"]


async def test_project_client_fails_closed_without_authentication() -> None:
    client = HttpProjectClient("https://apim.example.test/FunctionProjectApp")

    try:
        await client.get_project_context_batch("tenant-1", ["project-1"])
    except ValueError as exc:
        assert str(exc) == "Project service authentication is not configured."
    else:
        raise AssertionError("Unauthenticated project hydration should fail closed.")
