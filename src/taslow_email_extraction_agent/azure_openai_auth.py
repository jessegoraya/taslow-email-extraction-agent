from __future__ import annotations

from typing import Protocol

from azure.core.credentials import AccessToken
from azure.identity.aio import DefaultAzureCredential

from taslow_email_extraction_agent.config import Settings

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class AsyncTokenCredential(Protocol):
    async def get_token(self, *scopes: str) -> AccessToken:
        """Return an access token for the requested Azure resource scopes."""


class AzureOpenAIRequestAuthenticator:
    """Produces API-key or Microsoft Entra headers for Azure OpenAI requests."""

    def __init__(self, settings: Settings, credential: AsyncTokenCredential | None = None) -> None:
        self._settings = settings
        self._credential = credential

    @property
    def is_configured(self) -> bool:
        if (
            not self._settings.azure_openai_endpoint
            or not self._settings.azure_ai_model_deployment_name
        ):
            return False
        return (
            bool(self._settings.azure_openai_api_key)
            if self._settings.azure_openai_auth_mode == "api-key"
            else True
        )

    async def get_headers(self) -> dict[str, str]:
        if self._settings.azure_openai_auth_mode == "api-key":
            api_key = self._settings.azure_openai_api_key
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required when auth mode is api-key.")
            return {"api-key": api_key}

        if self._credential is None:
            self._credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        token = await self._credential.get_token(COGNITIVE_SERVICES_SCOPE)
        return {"Authorization": f"Bearer {token.token}"}
