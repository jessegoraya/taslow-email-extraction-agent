from __future__ import annotations

from pathlib import Path

from azure.core.credentials import AccessToken

from taslow_email_extraction_agent.azure_openai_auth import (
    COGNITIVE_SERVICES_SCOPE,
    AzureOpenAIRequestAuthenticator,
)
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.hosted_agent import process_email_invocation


class FakeCredential:
    def __init__(self) -> None:
        self.scopes: tuple[str, ...] | None = None

    async def get_token(self, *scopes: str) -> AccessToken:
        self.scopes = scopes
        return AccessToken("test-token", 4_102_444_800)


async def test_entra_authentication_requires_no_openai_key() -> None:
    credential = FakeCredential()
    settings = Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_ai_model_deployment_name="gpt-5.4-mini",
        azure_openai_auth_mode="entra",
    )

    authenticator = AzureOpenAIRequestAuthenticator(settings, credential=credential)

    assert authenticator.is_configured is True
    assert await authenticator.get_headers() == {"Authorization": "Bearer test-token"}
    assert credential.scopes == (COGNITIVE_SERVICES_SCOPE,)


async def test_hosted_invocation_runs_existing_email_workflow(base_request, services) -> None:
    status_code, body = await process_email_invocation(
        base_request.model_dump(by_alias=True, mode="json"),
        services=services,
        correlation_id="hosted-agent-test",
    )

    assert status_code == 200
    assert body["status"] == "tasks_ready"
    assert body["tasks"]


async def test_hosted_invocation_rejects_non_object_payload() -> None:
    status_code, body = await process_email_invocation(["not", "an", "email"])

    assert status_code == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_foundry_host_entrypoint_declares_invocations_contract() -> None:
    entrypoint = Path(__file__).parents[1] / "main.py"
    source = entrypoint.read_text(encoding="utf-8")
    deployment_manifest = Path(__file__).parents[1] / "azure.yaml"
    deployment_source = deployment_manifest.read_text(encoding="utf-8")

    assert "InvocationAgentServerHost()" in source
    assert "@app.invoke_handler" in source
    assert '"/readiness"' not in source
    assert "codeConfiguration:" in deployment_source
    assert "runtime: python_3_13" in deployment_source
    assert "protocol: invocations" in deployment_source
    assert 'value: "false"' in deployment_source
