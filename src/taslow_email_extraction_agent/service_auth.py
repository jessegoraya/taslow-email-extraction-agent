from __future__ import annotations

from azure.identity.aio import DefaultAzureCredential

from taslow_email_extraction_agent.azure_openai_auth import AsyncTokenCredential


class ManagedIdentityRequestAuthenticator:
    """Produces an Entra bearer token for a configured service audience."""

    def __init__(
        self,
        token_scope: str,
        credential: AsyncTokenCredential | None = None,
    ) -> None:
        self._token_scope = token_scope.strip()
        self._credential = credential

    async def get_headers(self) -> dict[str, str]:
        if not self._token_scope:
            raise ValueError("A service token scope is required for managed-identity auth.")
        if self._credential is None:
            self._credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True
            )
        token = await self._credential.get_token(self._token_scope)
        return {"Authorization": f"Bearer {token.token}"}
