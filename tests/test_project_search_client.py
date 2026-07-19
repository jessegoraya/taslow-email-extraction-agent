from __future__ import annotations

import json

import httpx
import pytest
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError

from taslow_email_extraction_agent.azure_openai_auth import COGNITIVE_SERVICES_SCOPE
from taslow_email_extraction_agent.azure_search_auth import AZURE_SEARCH_SCOPE
from taslow_email_extraction_agent.clients.project_search_client import (
    AzureProjectSearchClient,
    ProjectSearchUnavailable,
)
from taslow_email_extraction_agent.config import Settings


class FakeCredential:
    def __init__(self, *, failing_scope: str | None = None) -> None:
        self.failing_scope = failing_scope
        self.scopes: list[str] = []

    async def get_token(self, *scopes: str) -> AccessToken:
        scope = scopes[0]
        self.scopes.append(scope)
        if scope == self.failing_scope:
            raise ClientAuthenticationError(message=f"Token unavailable for {scope}")
        token = "search-token" if scope == AZURE_SEARCH_SCOPE else "openai-token"
        return AccessToken(token, 4_102_444_800)


def _settings() -> Settings:
    return Settings(
        azure_search_endpoint="https://taslow-test.search.windows.net",
        azure_search_index_name="taslow-project-scope-v1",
        azure_openai_endpoint="https://taslow-test.cognitiveservices.azure.com",
        azure_openai_auth_mode="entra",
        azure_openai_embedding_deployment="text-embedding-3-small",
    )


async def test_search_and_embedding_requests_use_entra_tokens_and_tenant_filter() -> None:
    credential = FakeCredential()
    observed_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(request)
        if request.url.path.endswith("/embeddings"):
            assert request.headers["Authorization"] == "Bearer openai-token"
            assert "api-key" not in request.headers
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

        assert request.url.path.endswith("/docs/search")
        assert request.headers["Authorization"] == "Bearer search-token"
        assert "api-key" not in request.headers
        payload = json.loads(request.content)
        assert "tenantId eq 'tenant-1'" in payload["filter"]
        assert "entityType eq 'project'" in payload["filter"]
        assert payload["vectorFilterMode"] == "preFilter"
        return httpx.Response(
            200,
            json={
                "value": [
                    {
                        "projectId": "project-1",
                        "scopeId": None,
                        "tenantId": "tenant-1",
                        "projectStatus": "Active",
                        "searchStatus": "active",
                        "isArchived": False,
                        "@search.score": 0.92,
                    }
                ]
            },
        )

    client = AzureProjectSearchClient(
        _settings(),
        credential=credential,
        transport=httpx.MockTransport(handler),
    )

    candidates = await client.search_projects("tenant-1", "electrical review")

    assert [candidate.project_id for candidate in candidates] == ["project-1"]
    assert credential.scopes == [COGNITIVE_SERVICES_SCOPE, AZURE_SEARCH_SCOPE]
    assert len(observed_requests) == 2


async def test_search_token_failure_is_retryable_and_sends_no_search_request() -> None:
    credential = FakeCredential(failing_scope=AZURE_SEARCH_SCOPE)
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    client = AzureProjectSearchClient(
        _settings(),
        credential=credential,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProjectSearchUnavailable, match="Azure AI Search query failed"):
        await client.search_projects("tenant-1", "electrical review")

    assert requested_paths == [
        "/openai/deployments/text-embedding-3-small/embeddings"
    ]
    assert credential.scopes == [COGNITIVE_SERVICES_SCOPE, AZURE_SEARCH_SCOPE]


async def test_embedding_token_failure_is_retryable_and_sends_no_request() -> None:
    credential = FakeCredential(failing_scope=COGNITIVE_SERVICES_SCOPE)
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(500)

    client = AzureProjectSearchClient(
        _settings(),
        credential=credential,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProjectSearchUnavailable, match="embedding generation failed"):
        await client.search_projects("tenant-1", "electrical review")

    assert requested_paths == []
    assert credential.scopes == [COGNITIVE_SERVICES_SCOPE]


def test_search_client_requires_no_api_key_in_entra_mode() -> None:
    settings = _settings()

    client = AzureProjectSearchClient(settings, credential=FakeCredential())
    aliases = {field.alias for field in Settings.model_fields.values()}

    assert client is not None
    assert "azure_search_api_key" not in Settings.model_fields
    assert "AZURE_SEARCH_API_KEY" not in aliases
