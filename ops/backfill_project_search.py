from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx
from azure.cosmos import CosmosClient

DATABASE_NAME = "bloomskyHealth"
PROJECT_CONTAINER_NAME = "Project"
SCHEMA_VERSION = "project-scope-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Project/Scope docs into Azure AI Search."
    )
    parser.add_argument("--tenant-id", default=os.getenv("TENANT_ID"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = _load_settings()
    projects = list(_read_projects(settings["cosmos_connection"], args.tenant_id))
    documents = []
    for project in projects:
        documents.extend(await _build_documents(project, settings))

    summary = {
        "tenantId": args.tenant_id,
        "projectsRead": len(projects),
        "documentsPrepared": len(documents),
        "documentsUploaded": 0,
        "documentsInactivated": sum(1 for doc in documents if doc["searchStatus"] == "inactive"),
        "dryRun": args.dry_run,
        "indexName": settings["search_index"],
    }

    if not args.dry_run and documents:
        summary["documentsUploaded"] = await _upload_documents(settings, documents, args.batch_size)

    print(json.dumps(summary, indent=2))


def _load_settings() -> dict[str, str | int]:
    required = {
        "cosmos_connection": os.getenv("COSMOSDB_CONNECTION")
        or os.getenv("PROJECT_COSMOS_CONNECTION_STRING"),
        "search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "search_index": os.getenv("AZURE_SEARCH_INDEX_NAME", "taslow-project-scope-v1"),
        "search_api_key": os.getenv("AZURE_SEARCH_API_KEY"),
        "search_api_version": os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01"),
        "openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "embedding_deployment": os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        ),
        "embedding_api_version": os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01"),
        "embedding_dimensions": int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")),
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    return required


def _read_projects(cosmos_connection: str, tenant_id: str | None) -> list[dict[str, Any]]:
    client = CosmosClient.from_connection_string(cosmos_connection)
    container = client.get_database_client(DATABASE_NAME).get_container_client(
        PROJECT_CONTAINER_NAME
    )
    if tenant_id:
        query = "SELECT * FROM c WHERE c.tenantID = @tenantId OR c.tenantid = @tenantId"
        parameters = [{"name": "@tenantId", "value": tenant_id}]
    else:
        query = "SELECT * FROM c"
        parameters = []

    return list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=tenant_id is None,
            partition_key=tenant_id if tenant_id else None,
        )
    )


async def _build_documents(
    project: dict[str, Any], settings: dict[str, str | int]
) -> list[dict[str, Any]]:
    tenant_id = _first(project, "tenantID", "tenantid", "TenantID")
    project_id = _first(project, "id", "ProjectID", "projectId")
    project_name = _first(project, "ProjectName", "projectName") or ""
    description = _first(project, "ProjectDescription", "projectDescription") or ""
    project_status = _first(project, "ProjectStatus", "projectStatus") or ""
    is_archived = bool(_first(project, "isArchived", "IsArchived") or project_status == "Archived")
    last_modified = _date_or_now(_first(project, "lastmodifieddate", "LastModifiedDate"))

    documents = []
    project_content = _join(project_name, description)
    if project_content:
        documents.append(
            await _search_document(
                settings=settings,
                tenant_id=tenant_id,
                entity_type="project",
                project_id=project_id,
                scope_id=None,
                source_id=project_id,
                title=project_name,
                content=project_content,
                project_status=project_status,
                search_status=_search_status(project_status, is_archived),
                is_archived=is_archived,
                last_modified=last_modified,
                metadata={"projectType": _first(project, "ProjectType", "projectType")},
            )
        )

    for scope in project.get("ProjectScopes") or project.get("projectScopes") or []:
        scope_id = _first(scope, "ScopeID", "scopeId", "scopeid")
        if not scope_id:
            continue
        title = _first(scope, "ProjectScopeAreaTitle", "projectScopeAreaTitle") or ""
        content = _first(scope, "ProjectScopeArea", "projectScopeArea") or ""
        scope_archived = bool(_first(scope, "isArchived", "isarchived") or is_archived)
        scope_content = _join(title, content)
        if not scope_content:
            continue
        documents.append(
            await _search_document(
                settings=settings,
                tenant_id=tenant_id,
                entity_type="scope",
                project_id=project_id,
                scope_id=scope_id,
                source_id=scope_id,
                title=title,
                content=scope_content,
                project_status=project_status,
                search_status=_search_status(project_status, scope_archived),
                is_archived=scope_archived,
                last_modified=last_modified,
                metadata={"groupTaskSetId": _first(scope, "GroupTaskSetID", "groupTaskSetId")},
            )
        )

    return documents


async def _search_document(
    settings: dict[str, str | int],
    tenant_id: str,
    entity_type: str,
    project_id: str,
    scope_id: str | None,
    source_id: str,
    title: str,
    content: str,
    project_status: str,
    search_status: str,
    is_archived: bool,
    last_modified: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    content_hash = _content_hash(
        content,
        project_status,
        search_status,
        str(settings["embedding_deployment"]),
        int(settings["embedding_dimensions"]),
        SCHEMA_VERSION,
    )
    return {
        "@search.action": "mergeOrUpload",
        "id": f"{tenant_id}_{entity_type}_{source_id}",
        "tenantId": tenant_id,
        "entityType": entity_type,
        "projectId": project_id,
        "scopeId": scope_id,
        "sourceId": source_id,
        "title": title,
        "content": content,
        "contentHash": content_hash,
        "projectStatus": project_status,
        "searchStatus": search_status,
        "isArchived": is_archived,
        "lastModifiedDate": last_modified,
        "embeddingModelId": settings["embedding_deployment"],
        "embeddingDimensions": settings["embedding_dimensions"],
        "schemaVersion": SCHEMA_VERSION,
        "metadata": json.dumps(metadata, separators=(",", ":")),
        "contentVector": await _embed(content, settings),
    }


async def _embed(text: str, settings: dict[str, str | int]) -> list[float]:
    endpoint = str(settings["openai_endpoint"]).rstrip("/")
    deployment = settings["embedding_deployment"]
    api_version = settings["embedding_api_version"]
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            headers={
                "api-key": str(settings["openai_api_key"]),
                "Content-Type": "application/json",
            },
            json={"input": text},
        )
        response.raise_for_status()
        body = response.json()
    return body["data"][0]["embedding"]


async def _upload_documents(
    settings: dict[str, str | int], documents: list[dict[str, Any]], batch_size: int
) -> int:
    endpoint = str(settings["search_endpoint"]).rstrip("/")
    index_name = settings["search_index"]
    api_version = settings["search_api_version"]
    url = f"{endpoint}/indexes/{index_name}/docs/index?api-version={api_version}"
    uploaded = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = await client.post(
                url,
                headers={
                    "api-key": str(settings["search_api_key"]),
                    "Content-Type": "application/json",
                },
                json={"value": batch},
            )
            response.raise_for_status()
            uploaded += len(batch)
    return uploaded


def _search_status(project_status: str, is_archived: bool) -> str:
    return "active" if project_status == "Active" and not is_archived else "inactive"


def _content_hash(*parts: str | int) -> str:
    normalized = "\n".join(str(part or "").strip() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _join(*parts: str) -> str:
    cleaned = []
    for part in parts:
        text = _text(part)
        if text:
            cleaned.append(text)
    return " ".join(cleaned)


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text(item) for item in value if _text(item)).strip()
    if isinstance(value, dict):
        for key in ("name", "Name", "title", "Title", "value", "Value", "text", "Text"):
            if key in value:
                return _text(value[key])
    return ""


def _first(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def _date_or_now(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        parsed = value.strip()
        if parsed.endswith("Z"):
            return parsed
        return parsed.replace("+00:00", "Z")
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    asyncio.run(main())
