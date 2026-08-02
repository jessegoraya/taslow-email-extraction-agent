from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

import httpx

from taslow_email_extraction_agent.azure_openai_auth import AsyncTokenCredential
from taslow_email_extraction_agent.models import AssociatedPerson, ProjectContext, ProjectScope
from taslow_email_extraction_agent.service_auth import ManagedIdentityRequestAuthenticator


class ProjectCandidateDiscoveryUnavailable(RuntimeError):
    """Participant-based Project candidate discovery could not be completed."""


class ProjectClient(Protocol):
    async def get_active_projects(self, tenant_id: str) -> list[ProjectContext]:
        """Return active tenant projects with enough context for candidate scoring."""

    async def get_project_detail(self, tenant_id: str, project_id: str) -> ProjectContext | None:
        """Return hydrated Project detail from the Project source of truth."""

    async def get_project_context_batch(
        self, tenant_id: str, project_ids: list[str]
    ) -> list[ProjectContext]:
        """Return service-friendly Project context for selected Project IDs."""

    async def get_participant_project_candidates(
        self, tenant_id: str, participant_emails: list[str]
    ) -> list[str]:
        """Return active Project IDs containing at least one email participant."""


class InMemoryProjectClient:
    def __init__(self, projects: list[ProjectContext] | None = None) -> None:
        self._projects = projects or []

    async def get_active_projects(self, tenant_id: str) -> list[ProjectContext]:
        return self._projects

    async def get_project_detail(self, tenant_id: str, project_id: str) -> ProjectContext | None:
        for project in self._projects:
            if project.project_id == project_id:
                return project
        return None

    async def get_project_context_batch(
        self, tenant_id: str, project_ids: list[str]
    ) -> list[ProjectContext]:
        wanted = set(project_ids)
        return [project for project in self._projects if project.project_id in wanted]

    async def get_participant_project_candidates(
        self, tenant_id: str, participant_emails: list[str]
    ) -> list[str]:
        participants = {email.strip().lower() for email in participant_emails if email.strip()}
        matches = []
        for project in self._projects:
            project_emails = {person.email for person in project.people if person.email}
            overlap_count = len(participants & project_emails)
            if overlap_count:
                matches.append((project.project_id, overlap_count))
        ordered_matches = sorted(matches, key=lambda item: (-item[1], item[0]))
        return [project_id for project_id, _ in ordered_matches]


class HttpProjectClient:
    """Project service client using the existing Taslow Project endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        token_scope: str | None = None,
        *,
        credential: AsyncTokenCredential | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._authenticator = (
            ManagedIdentityRequestAuthenticator(
                token_scope,
                credential=credential,
            )
            if token_scope
            else None
        )
        self._transport = transport
        host = urlparse(self._base_url).netloc.lower()
        self._path_prefix = "/api" if host.endswith("azurewebsites.net") else ""

    async def get_active_projects(self, tenant_id: str) -> list[ProjectContext]:
        headers = await self._get_headers()

        async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
            response = await client.get(
                self._url(f"/projects/active/{tenant_id}"),
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, dict):
            rows = payload.get("projects") or payload.get("data") or []
        else:
            rows = payload

        return [self._map_project(row) for row in rows]

    async def get_project_detail(self, tenant_id: str, project_id: str) -> ProjectContext | None:
        headers = await self._get_headers()

        async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
            response = await client.get(
                self._url(f"/projects/{tenant_id}/{project_id}/detail"),
                headers=headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()

        return self._map_project(payload)

    async def get_project_context_batch(
        self, tenant_id: str, project_ids: list[str]
    ) -> list[ProjectContext]:
        if not project_ids:
            return []

        headers = await self._get_headers()

        payload = {
            "tenantId": tenant_id,
            "projectIds": project_ids,
            "includeScopes": True,
            "includeAssociatedPeople": True,
            "includeAssociatedManagers": True,
        }

        async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
            response = await client.post(
                self._url("/internal/projects/agent-context/batch"),
                headers=headers,
                json=payload,
            )
            if response.status_code in {404, 405}:
                return await self._fallback_project_detail_batch(tenant_id, project_ids)
            response.raise_for_status()
            body = response.json()

        rows = body if isinstance(body, list) else body.get("projects", [])
        return [self._map_project(row) for row in rows]

    async def get_participant_project_candidates(
        self, tenant_id: str, participant_emails: list[str]
    ) -> list[str]:
        normalized_emails = sorted(
            {email.strip().lower() for email in participant_emails if email.strip()}
        )
        if not normalized_emails:
            return []

        headers = await self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
                response = await client.post(
                    self._url("/internal/projects/participant-candidates"),
                    headers=headers,
                    json={"tenantId": tenant_id, "participantEmails": normalized_emails},
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProjectCandidateDiscoveryUnavailable(
                "Participant Project candidate discovery failed."
            ) from exc

        if not isinstance(body, (dict, list)):
            raise ProjectCandidateDiscoveryUnavailable(
                "Participant Project candidate response is invalid."
            )
        rows = body if isinstance(body, list) else body.get("projects", [])
        return [
            str(row.get("projectId", "")).strip()
            for row in rows
            if str(row.get("projectId", "")).strip()
        ]

    async def _fallback_project_detail_batch(
        self, tenant_id: str, project_ids: list[str]
    ) -> list[ProjectContext]:
        projects: list[ProjectContext] = []
        for project_id in project_ids:
            project = await self.get_project_detail(tenant_id, project_id)
            if project:
                projects.append(project)
        return projects

    def _map_project(self, row: dict) -> ProjectContext:
        scopes = [
            ProjectScope(
                scopeId=scope.get("ScopeID") or scope.get("scopeId") or scope.get("scopeID") or "",
                title=scope.get("ProjectScopeAreaTitle")
                or scope.get("scopeTitle")
                or scope.get("title")
                or "",
                description=scope.get("ProjectScopeArea")
                or scope.get("scopeDescription")
                or scope.get("description")
                or "",
                embeddings=scope.get("ProjectScopeAreaEmbeddings")
                or scope.get("projectScopeAreaEmbeddings")
                or [],
                groupTaskSetId=scope.get("GroupTaskSetID") or scope.get("groupTaskSetId"),
            )
            for scope in row.get("ProjectScopes", row.get("scopes", [])) or []
        ]
        people = [
            self._map_person(person)
            for person in row.get("AssociatedPeople", row.get("associatedPeople", [])) or []
        ]
        managers = [
            self._map_person(person)
            for person in row.get("AssociatedManagers", row.get("associatedManagers", [])) or []
        ]
        return ProjectContext(
            projectId=row.get("id") or row.get("ProjectID") or row.get("projectId") or "",
            projectName=row.get("ProjectName") or row.get("projectName") or "",
            description=row.get("ProjectDescription")
            or row.get("projectDescription")
            or row.get("description")
            or "",
            descVector=row.get("DescVector") or row.get("descVector") or [],
            clientDomains=row.get("ClientDomains") or row.get("clientDomains") or [],
            associatedPeople=people,
            associatedManagers=managers,
            scopes=scopes,
        )

    @staticmethod
    def _map_person(row: dict) -> AssociatedPerson:
        return AssociatedPerson(
            personId=str(row.get("AssociatedPersonID") or row.get("personId") or ""),
            name=row.get("PersonName") or row.get("displayName") or row.get("name") or "",
            aliases=row.get("PersonAliases") or row.get("aliases") or "",
            email=row.get("PersonEmail") or row.get("email") or "",
            role=row.get("Role") or row.get("role") or "",
        )

    def _url(self, path: str) -> str:
        return f"{self._base_url}{self._path_prefix}{path}"

    async def _get_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"x-functions-key": self._api_key}
        if self._authenticator:
            return await self._authenticator.get_headers()
        raise ValueError("Project service authentication is not configured.")
