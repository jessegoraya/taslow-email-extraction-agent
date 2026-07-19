from __future__ import annotations

from azure.identity.aio import DefaultAzureCredential

from taslow_email_extraction_agent.azure_openai_auth import AsyncTokenCredential

AZURE_SEARCH_SCOPE = "https://search.azure.com/.default"


class AzureSearchRequestAuthenticator:
    """Produces Microsoft Entra bearer headers for Azure AI Search requests."""

    def __init__(self, credential: AsyncTokenCredential | None = None) -> None:
        self._credential = credential

    async def get_headers(self) -> dict[str, str]:
        if self._credential is None:
            self._credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True
            )
        token = await self._credential.get_token(AZURE_SEARCH_SCOPE)
        return {"Authorization": f"Bearer {token.token}"}
