from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from taslow_email_extraction_agent.config import Settings


class ProjectSearchUnavailable(RuntimeError):
    """Raised when Azure AI Search or query embedding generation is unavailable."""


@dataclass(slots=True)
class SearchCandidate:
    project_id: str
    scope_id: str | None
    score: float
    rank: int = 0
    score_raw: float | None = None
    score_margin: float = 0.0


class ProjectSearchClient(Protocol):
    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        """Return active tenant Project candidates from the Search index."""

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        """Return active Scope candidates for a selected Project."""


class AzureProjectSearchClient:
    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in [
                ("AZURE_SEARCH_ENDPOINT", settings.azure_search_endpoint),
                ("AZURE_SEARCH_INDEX_NAME", settings.azure_search_index_name),
                ("AZURE_SEARCH_API_KEY", settings.azure_search_api_key),
                ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
                ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Azure project search settings: {', '.join(missing)}")

        self._search_endpoint = settings.azure_search_endpoint.rstrip("/")
        self._search_index_name = settings.azure_search_index_name
        self._search_api_key = settings.azure_search_api_key
        self._search_api_version = settings.azure_search_api_version
        self._openai_endpoint = settings.azure_openai_endpoint.rstrip("/")
        self._openai_api_key = settings.azure_openai_api_key
        self._embedding_deployment = settings.azure_openai_embedding_deployment
        self._embedding_api_version = settings.azure_openai_embedding_api_version
        self._project_top_k = settings.project_search_top_k
        self._scope_top_k = settings.scope_search_top_k

    async def search_projects(self, tenant_id: str, query_text: str) -> list[SearchCandidate]:
        filter_expression = (
            f"tenantId eq '{_escape_odata(tenant_id)}' "
            "and entityType eq 'project' "
            "and projectStatus eq 'Active' "
            "and searchStatus eq 'active' "
            "and isArchived eq false"
        )
        return await self._search(query_text, filter_expression, self._project_top_k)

    async def search_scopes(
        self, tenant_id: str, project_id: str, query_text: str
    ) -> list[SearchCandidate]:
        filter_expression = (
            f"tenantId eq '{_escape_odata(tenant_id)}' "
            "and entityType eq 'scope' "
            f"and projectId eq '{_escape_odata(project_id)}' "
            "and projectStatus eq 'Active' "
            "and searchStatus eq 'active' "
            "and isArchived eq false"
        )
        return await self._search(query_text, filter_expression, self._scope_top_k)

    async def _search(
        self, query_text: str, filter_expression: str, top_k: int
    ) -> list[SearchCandidate]:
        vector = await self._embed_query(query_text)
        payload = {
            "count": False,
            "filter": filter_expression,
            "select": (
                "id,tenantId,entityType,projectId,scopeId,sourceId,"
                "projectStatus,searchStatus,isArchived"
            ),
            "top": top_k,
            "vectorFilterMode": "preFilter",
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vector,
                    "fields": "contentVector",
                    "k": top_k,
                }
            ],
        }
        url = (
            f"{self._search_endpoint}/indexes/{self._search_index_name}/docs/search"
            f"?api-version={self._search_api_version}"
        )
        headers = {"api-key": self._search_api_key, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise ProjectSearchUnavailable("Azure AI Search query failed.") from exc

        rows = [
            SearchCandidate(
                project_id=row.get("projectId") or "",
                scope_id=row.get("scopeId"),
                score=float(row.get("@search.score") or 0.0),
                score_raw=float(row.get("@search.score") or 0.0),
            )
            for row in body.get("value", [])
            if row.get("projectId")
            and row.get("tenantId")
            and row.get("projectStatus") == "Active"
            and row.get("searchStatus") == "active"
            and row.get("isArchived") is False
        ]
        return _normalize_candidates(rows)

    async def _embed_query(self, query_text: str) -> list[float]:
        url = (
            f"{self._openai_endpoint}/openai/deployments/{self._embedding_deployment}"
            f"/embeddings?api-version={self._embedding_api_version}"
        )
        headers = {"api-key": self._openai_api_key, "Content-Type": "application/json"}
        payload = {"input": query_text}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise ProjectSearchUnavailable("Azure OpenAI embedding generation failed.") from exc

        try:
            return body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProjectSearchUnavailable("Azure OpenAI embedding response was invalid.") from exc


def _escape_odata(value: str) -> str:
    return value.replace("'", "''")


def _normalize_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    if not candidates:
        return []

    raw_scores = [
        candidate.score_raw if candidate.score_raw is not None else candidate.score
        for candidate in candidates
    ]
    top_score = max(raw_scores) if raw_scores else 0.0
    second_score = sorted(raw_scores, reverse=True)[1] if len(raw_scores) > 1 else 0.0
    margin = max(0.0, top_score - second_score)

    normalized: list[SearchCandidate] = []
    for index, candidate in enumerate(candidates, start=1):
        raw = candidate.score_raw if candidate.score_raw is not None else candidate.score
        # Azure AI Search vector scores are useful for ranking, but not a calibrated confidence.
        # Keep the top result near its original score and preserve relative distance.
        relative = raw / top_score if top_score > 0 else 0.0
        normalized_score = max(0.0, min(1.0, (raw * 0.70) + (relative * 0.30)))
        normalized.append(
            SearchCandidate(
                project_id=candidate.project_id,
                scope_id=candidate.scope_id,
                score=round(normalized_score, 4),
                rank=index,
                score_raw=round(raw, 4),
                score_margin=round(margin, 4) if index == 1 else 0.0,
            )
        )
    return normalized
