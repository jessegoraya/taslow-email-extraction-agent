from __future__ import annotations

from typing import Protocol

import httpx

from taslow_email_extraction_agent.models import AssociatedPerson, ProjectContext, ProjectScope


class ProjectClient(Protocol):
    async def get_active_projects(self, tenant_id: str) -> list[ProjectContext]:
        """Return active tenant projects with enough context for candidate scoring."""

    async def get_project_detail(self, tenant_id: str, project_id: str) -> ProjectContext | None:
        """Return hydrated Project detail from the Project source of truth."""

    async def get_project_context_batch(
        self, tenant_id: str, project_ids: list[str]
    ) -> list[ProjectContext]:
        """Return service-friendly Project context for selected Project IDs."""


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


class HttpProjectClient:
    """Project service client using the existing Taslow Project endpoints."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def get_active_projects(self, tenant_id: str) -> list[ProjectContext]:
        headers = {}
        if self._api_key:
            headers["x-functions-key"] = self._api_key

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{self._base_url}/api/projects/active/{tenant_id}",
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
        headers = {}
        if self._api_key:
            headers["x-functions-key"] = self._api_key

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{self._base_url}/api/projects/{tenant_id}/{project_id}/detail",
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

        headers = {}
        if self._api_key:
            headers["x-functions-key"] = self._api_key

        payload = {
            "tenantId": tenant_id,
            "projectIds": project_ids,
            "includeScopes": True,
            "includeAssociatedPeople": True,
            "includeAssociatedManagers": True,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{self._base_url}/api/internal/projects/agent-context/batch",
                headers=headers,
                json=payload,
            )
            if response.status_code in {404, 405}:
                return await self._fallback_project_detail_batch(tenant_id, project_ids)
            response.raise_for_status()
            body = response.json()

        rows = body if isinstance(body, list) else body.get("projects", [])
        return [self._map_project(row) for row in rows]

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
