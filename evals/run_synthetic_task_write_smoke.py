from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from azure.cosmos import CosmosClient


ZERO_GUID = "00000000-0000-0000-0000-000000000000"
SOURCE_SYSTEM = "taslow-ai-synthetic-evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled synthetic task-write smoke test through Logic App."
    )
    parser.add_argument("--stage1-run-dir", required=True, type=Path)
    parser.add_argument("--project-context", required=True, type=Path)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--synthetic-run-id", required=True)
    parser.add_argument("--logic-app-endpoint", default=os.getenv("TASLOW_TASK_WRITE_LOGIC_APP_ENDPOINT"))
    parser.add_argument("--project-callback-base-url", default=os.getenv("PROJECT_SCOPE_LINK_CALLBACK_BASE_URL"))
    parser.add_argument(
        "--project-callback-function-key",
        default=os.getenv("PROJECT_SCOPE_LINK_CALLBACK_FUNCTION_KEY"),
    )
    parser.add_argument(
        "--project-function-base-url",
        default=os.getenv("PROJECT_FUNCTION_BASE_URL") or os.getenv("PROJECT_SCOPE_LINK_CALLBACK_BASE_URL"),
    )
    parser.add_argument(
        "--project-agent-context-function-key",
        default=os.getenv("PROJECT_AGENT_CONTEXT_FUNCTION_KEY") or os.getenv("PROJECT_FUNCTION_KEY"),
    )
    parser.add_argument("--task-function-base-url", default=os.getenv("TASK_FUNCTION_BASE_URL"))
    parser.add_argument("--task-function-key", default=os.getenv("TASK_FUNCTION_KEY"))
    parser.add_argument("--max-task-writes", type=int, default=5)
    parser.add_argument(
        "--exclude-source-lines",
        default="",
        help="Comma-separated sourceLine values to skip, useful after a partial smoke write.",
    )
    parser.add_argument(
        "--exclude-stage2-run-dir",
        action="append",
        default=[],
        type=Path,
        help="Prior Stage 2 run folder whose verified sourceLine values should be skipped.",
    )
    parser.add_argument("--write-tasks", action="store_true")
    parser.add_argument("--allow-create-gts", action="store_true")
    parser.add_argument(
        "--existing-gts-write-mode",
        choices=["logic-app", "direct-task-api", "direct-cosmos"],
        default="logic-app",
        help="How to add tasks when the Project scope already has a groupTaskSetId.",
    )
    parser.add_argument(
        "--cosmos-connection-string",
        default=os.getenv("COSMOSDB_CONNECTION") or os.getenv("CosmosDBConnection"),
    )
    parser.add_argument("--skip-live-project-context-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--max-write-retries", type=int, default=3)
    parser.add_argument("--retry-base-delay-seconds", type=float, default=10)
    parser.add_argument("--inter-write-delay-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not args.write_tasks and not args.dry_run:
        raise SystemExit("Refusing to run: pass --write-tasks or --dry-run explicitly.")
    if args.write_tasks and not args.logic_app_endpoint:
        raise SystemExit("TASLOW_TASK_WRITE_LOGIC_APP_ENDPOINT or --logic-app-endpoint is required.")
    if args.write_tasks and args.allow_create_gts:
        if not args.project_callback_base_url or not args.project_callback_function_key:
            raise SystemExit(
                "Creating GroupTaskSets requires PROJECT_SCOPE_LINK_CALLBACK_BASE_URL "
                "and PROJECT_SCOPE_LINK_CALLBACK_FUNCTION_KEY."
            )
    if args.write_tasks and not args.skip_live_project_context_refresh:
        if not args.project_function_base_url or not args.project_agent_context_function_key:
            raise SystemExit(
                "Write mode requires live Project context refresh. Set PROJECT_FUNCTION_BASE_URL "
                "and PROJECT_AGENT_CONTEXT_FUNCTION_KEY, or pass --skip-live-project-context-refresh."
            )
    if args.write_tasks and args.existing_gts_write_mode == "direct-cosmos":
        if not args.cosmos_connection_string:
            raise SystemExit(
                "COSMOSDB_CONNECTION or --cosmos-connection-string is required for direct-cosmos mode."
            )
    args.exclude_source_lines = _parse_source_lines(args.exclude_source_lines)
    args.exclude_source_lines.update(_load_excluded_source_lines(args.exclude_stage2_run_dir))

    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{args.synthetic_run_id}"
    out_dir = args.stage1_run_dir / "stage2-task-write-smoke" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    requests_by_line = _load_jsonl_by_source_line(args.stage1_run_dir / "selected_requests.jsonl")
    responses_by_line = _load_jsonl_by_source_line(args.stage1_run_dir / "agent_responses.jsonl")
    scoring_by_line = _load_jsonl_by_source_line(args.stage1_run_dir / "scoring_details.jsonl")
    context = _load_project_context(args.project_context)
    refresh_report: dict[str, Any] = {
        "enabled": False,
        "reason": "not_configured_or_skipped",
    }

    if not args.skip_live_project_context_refresh and (
        args.project_function_base_url and args.project_agent_context_function_key
    ):
        async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
            refresh_report = await _refresh_project_context_from_project_app(
                client,
                context=context,
                base_url=args.project_function_base_url,
                function_key=args.project_agent_context_function_key,
                tenant_id=args.tenant_id,
            )

    candidates = _select_candidates(
        requests_by_line=requests_by_line,
        responses_by_line=responses_by_line,
        scoring_by_line=scoring_by_line,
        context=context,
        tenant_id=args.tenant_id,
        synthetic_run_id=args.synthetic_run_id,
        max_task_writes=args.max_task_writes,
        exclude_source_lines=args.exclude_source_lines,
        allow_create_gts=args.allow_create_gts,
        project_callback_base_url=args.project_callback_base_url,
        project_callback_function_key=args.project_callback_function_key,
    )
    _write_json(out_dir / "stage2_project_context_refresh.json", refresh_report)
    _write_jsonl(out_dir / "stage2_write_candidates.jsonl", candidates)
    _write_jsonl(out_dir / "stage2_task_write_requests.jsonl", [c["logicAppPayload"] for c in candidates])

    responses: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    if not args.dry_run:
        cosmos_container = (
            _cosmos_group_task_set_container(args.cosmos_connection_string)
            if args.existing_gts_write_mode == "direct-cosmos"
            else None
        )
        async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
            for candidate in candidates:
                before = await _fetch_group_task_set_for_mode(
                    client,
                    cosmos_container=cosmos_container,
                    base_url=args.task_function_base_url,
                    function_key=args.task_function_key,
                    group_task_set_id=candidate["groupTaskSetIdBeforeWrite"],
                    tenant_id=args.tenant_id,
                )
                duplicate_task = _find_duplicate_task(before, candidate)
                if duplicate_task:
                    write_response = _duplicate_skip_response(candidate)
                    responses.append(write_response)
                    verifications.append(
                        _verify_duplicate_skip(
                            candidate=candidate,
                            group_task_set=before,
                            write_response=write_response,
                            duplicate_task=duplicate_task,
                        )
                    )
                    continue
                if (
                    args.existing_gts_write_mode == "direct-task-api"
                    and candidate["groupTaskSetIdBeforeWrite"] != ZERO_GUID
                ):
                    write_response = await _post_add_group_task_direct(
                        client,
                        base_url=args.task_function_base_url,
                        function_key=args.task_function_key,
                        candidate=candidate,
                        max_retries=args.max_write_retries,
                        base_delay_seconds=args.retry_base_delay_seconds,
                    )
                elif (
                    args.existing_gts_write_mode == "direct-cosmos"
                    and candidate["groupTaskSetIdBeforeWrite"] != ZERO_GUID
                ):
                    write_response = await _append_group_task_cosmos(
                        cosmos_container,
                        candidate=candidate,
                        max_retries=args.max_write_retries,
                        base_delay_seconds=args.retry_base_delay_seconds,
                    )
                else:
                    write_response = await _post_logic_app_with_retries(
                        client,
                        args.logic_app_endpoint,
                        candidate["logicAppPayload"],
                        candidate["headers"],
                        max_retries=args.max_write_retries,
                        base_delay_seconds=args.retry_base_delay_seconds,
                    )
                responses.append(write_response)
                group_task_set_id = (
                    (write_response.get("json") or {}).get("groupTaskSetId")
                    or candidate["groupTaskSetIdBeforeWrite"]
                )
                after = await _fetch_group_task_set_for_mode(
                    client,
                    cosmos_container=cosmos_container,
                    base_url=args.task_function_base_url,
                    function_key=args.task_function_key,
                    group_task_set_id=group_task_set_id,
                    tenant_id=args.tenant_id,
                )
                verifications.append(
                    _verify_write(
                        candidate=candidate,
                        before=before,
                        after=after,
                        write_response=write_response,
                        group_task_set_id=group_task_set_id,
                    )
                )
                if args.inter_write_delay_seconds > 0:
                    await asyncio.sleep(args.inter_write_delay_seconds)

    summary = _summary(args, run_id, candidates, responses, verifications)
    _write_jsonl(out_dir / "stage2_task_write_responses.jsonl", responses)
    _write_jsonl(out_dir / "stage2_task_write_verification.jsonl", verifications)
    _write_json(out_dir / "stage2_task_write_summary.json", summary)
    _write_summary_md(out_dir / "stage2_task_write_summary.md", summary)
    print(json.dumps({"outputDir": str(out_dir), **summary}, indent=2))


def _select_candidates(
    *,
    requests_by_line: dict[int, dict[str, Any]],
    responses_by_line: dict[int, dict[str, Any]],
    scoring_by_line: dict[int, dict[str, Any]],
    context: dict[str, Any],
    tenant_id: str,
    synthetic_run_id: str,
    max_task_writes: int,
    exclude_source_lines: set[int],
    allow_create_gts: bool,
    project_callback_base_url: str | None,
    project_callback_function_key: str | None,
) -> list[dict[str, Any]]:
    used_create_scope_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for source_line in sorted(scoring_by_line):
        score = scoring_by_line[source_line]
        if source_line in exclude_source_lines:
            continue
        if not score.get("passed"):
            continue
        if score.get("actualStatus") != "tasks_ready":
            continue
        raw = responses_by_line.get(source_line) or {}
        response = raw.get("response") or {}
        request = requests_by_line.get(source_line) or {}
        if response.get("tenantId") != tenant_id or request.get("tenantId") != tenant_id:
            continue
        tasks = response.get("tasks") or []
        if len(tasks) != 1:
            continue
        task = tasks[0]
        project_id = task.get("projectId") or (response.get("projectMatch") or {}).get("projectId")
        scope_id = task.get("scopeId")
        if not project_id or not scope_id:
            continue
        project = context["projectsById"].get(project_id)
        scope = (project or {}).get("scopesById", {}).get(scope_id)
        if not project or not scope:
            continue
        group_task_set_id = scope.get("groupTaskSetId") or ""
        if not group_task_set_id and scope_id in used_create_scope_ids:
            continue
        if not group_task_set_id and not allow_create_gts:
            continue
        candidate_id = f"{synthetic_run_id}-source-{source_line}"
        logic_payload = _build_logic_app_payload(
            candidate_id=candidate_id,
            synthetic_run_id=synthetic_run_id,
            source_line=source_line,
            request=request,
            response=response,
            task=task,
            project=project,
            scope=scope,
            group_task_set_id=group_task_set_id,
            project_callback_base_url=project_callback_base_url,
            project_callback_function_key=project_callback_function_key,
        )
        candidates.append(
            {
                "candidateId": candidate_id,
                "sourceLine": source_line,
                "tenantId": tenant_id,
                "projectId": project_id,
                "projectName": project.get("projectName"),
                "scopeId": scope_id,
                "scopeTitle": scope.get("scopeTitle"),
                "groupTaskSetIdBeforeWrite": group_task_set_id or ZERO_GUID,
                "usesExistingGroupTaskSet": bool(group_task_set_id),
                "assigneeEmail": task.get("assigneeEmail"),
                "agentRunId": response.get("agentRunId"),
                "messageId": request.get("messageId"),
                "internetMessageId": request.get("internetMessageId"),
                "logicAppPayload": logic_payload,
                "headers": {
                    "Content-Type": "application/json",
                    "x-taslow-tenant-id": tenant_id,
                    "x-taslow-correlation-id": request.get("correlationId", ""),
                    "x-taslow-agent-run-id": response.get("agentRunId", ""),
                    "x-taslow-synthetic-run-id": synthetic_run_id,
                    "x-taslow-idempotency-key": _idempotency_key(tenant_id, request, task),
                },
            }
        )
        if not group_task_set_id:
            used_create_scope_ids.add(scope_id)
        if len(candidates) >= max_task_writes:
            break
    return candidates


def _build_logic_app_payload(
    *,
    candidate_id: str,
    synthetic_run_id: str,
    source_line: int,
    request: dict[str, Any],
    response: dict[str, Any],
    task: dict[str, Any],
    project: dict[str, Any],
    scope: dict[str, Any],
    group_task_set_id: str,
    project_callback_base_url: str | None,
    project_callback_function_key: str | None,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    due_date = task.get("dueDate") or "0001-01-01T00:00:00Z"
    group_task_id = str(uuid.uuid4())
    individual_task_set_id = str(uuid.uuid4())
    individual_task_id = str(uuid.uuid4())
    sender_email = request.get("from", {}).get("email", "")
    sender_name = request.get("from", {}).get("name", "")
    assignee_email = task.get("assigneeEmail", "")
    assignee_name = task.get("assigneeName", "")
    notes = (
        f"sourceSystem={SOURCE_SYSTEM}; syntheticRunId={synthetic_run_id}; "
        f"candidateId={candidate_id}; sourceLine={source_line}; "
        f"agentRunId={response.get('agentRunId')}; messageId={request.get('messageId')}; "
        f"internetMessageId={request.get('internetMessageId')}; "
        f"idempotencyKey={_idempotency_key(response.get('tenantId', ''), request, task)}"
    )
    group_task = {
        "_type": "GroupTask",
        "GroupTaskID": group_task_id,
        "GroupTaskTitle": _truncate(task.get("title") or "Synthetic email task", 180),
        "GroupTaskDescription": task.get("description") or task.get("title") or "",
        "GroupTaskStatus": "Open",
        "GroupTaskDueDate": [
            {
                "GroupTaskDueDateSequence": 1,
                "GroupTaskDueDate": due_date,
                "LastGroupTaskDueDate": due_date,
            }
        ],
        "GroupTaskClosedDate": "0001-01-01T00:00:00Z",
        "AssociatedDocuments": [],
        "AssociatedLOBItems": [],
        "AssoicatedDocuments": [],
        "AssoicatedLOBItems": [],
        "GroupTaskType": "Email Extracted Task",
        "GroupTaskStage": "Awaiting Assignment",
        "AssignorStakeholderGroup": {
            "AssignorStakeholderGroupID": _stable_guid(sender_email or sender_name or candidate_id),
            "AssignorStakeholderGroup": sender_name,
        },
        "AssigneeStakeholderGroup": [
            {
                "AssigneeStakeholderGroupID": _stable_guid(assignee_email or assignee_name or candidate_id),
                "AssigneeStakeholderGroup": assignee_name,
            }
        ],
        "GroupTaskNotes": notes,
        "FacilitiationComplete": False,
        "FacilitiationPreviouslyComplete": False,
        "CancellationSent": False,
        "ParentGroupTaskID": ZERO_GUID,
        "CreatedBy": SOURCE_SYSTEM,
        "CreatedDate": now,
        "LastModifiedBy": SOURCE_SYSTEM,
        "LastModifiedDate": now,
        "IndividualTaskSets": [
            {
                "IndividualTaskSetID": individual_task_set_id,
                "CreatedBy": SOURCE_SYSTEM,
                "CreatedDate": now,
                "IndividualTask": [
                    {
                        "IndividualTaskID": individual_task_id,
                        "IndividualTaskStatus": "Open",
                        "IndividualTaskTitle": _truncate(task.get("title") or "", 180),
                        "IndividualTaskType": "Email Extracted Task",
                        "IndividualTaskDescription": task.get("description") or task.get("title") or "",
                        "IndividualTaskNotes": notes,
                        "Priority": "Normal",
                        "AssignedPerson": assignee_email,
                        "AssociatedRole": assignee_name,
                        "PreviouslySent": False,
                        "IndividualTaskAssignedDate": now,
                        "IndividualTaskDueDate": due_date,
                        "IndividualTaskCancelledDate": "0001-01-01T00:00:00Z",
                        "IndividualTaskApprovalDecision": "",
                        "IndividualTaskCompletedDate": "0001-01-01T00:00:00Z",
                        "CreatedBy": SOURCE_SYSTEM,
                        "CreatedDate": now,
                    }
                ],
            }
        ],
    }
    payload = {
        "GroupTask": [group_task],
        "ProjectID": project["projectId"],
        "TenantID": response["tenantId"],
        "id": group_task_set_id or ZERO_GUID,
        "ScopeID": scope["scopeId"],
        "ProjectScopeAreaTitle": scope.get("scopeTitle", ""),
        "ProjectScopeArea": scope.get("scopeDescription", ""),
        "ProjectScopeAreaEmbeddings": [],
        "OrchestrationRunId": candidate_id,
    }
    if group_task_set_id:
        payload["grouptask"] = [group_task]
    else:
        payload["ProjectScopeLinkCallbackUrl"] = (
            f"{(project_callback_base_url or '').rstrip('/')}/projects/"
            f"{response['tenantId']}/{project['projectId']}/scopes/link-gts"
        )
        payload["ProjectScopeLinkSecret"] = project_callback_function_key or ""
    return payload


async def _refresh_project_context_from_project_app(
    client: httpx.AsyncClient,
    *,
    context: dict[str, Any],
    base_url: str,
    function_key: str,
    tenant_id: str,
) -> dict[str, Any]:
    project_ids = sorted(context.get("projectsById", {}).keys())
    if not project_ids:
        return {
            "enabled": True,
            "projectCount": 0,
            "updatedScopeLinkCount": 0,
            "reason": "no_projects_in_local_context",
        }

    url = f"{base_url.rstrip('/')}/internal/projects/agent-context/batch"
    response = await client.post(
        url,
        params={"code": function_key},
        json={
            "tenantId": tenant_id,
            "projectIds": project_ids,
            "includeScopes": True,
            "includeAssociatedPeople": False,
            "includeAssociatedManagers": False,
        },
    )
    response.raise_for_status()
    payload = response.json()
    updated_scope_links = 0
    live_scope_links = 0
    missing_projects: list[str] = []

    live_projects = {project.get("projectId"): project for project in payload.get("projects", [])}
    for project_id, local_project in context.get("projectsById", {}).items():
        live_project = live_projects.get(project_id)
        if not live_project:
            missing_projects.append(project_id)
            continue
        for live_scope in live_project.get("scopes") or []:
            scope_id = live_scope.get("scopeId")
            group_task_set_id = live_scope.get("groupTaskSetId") or ""
            if group_task_set_id:
                live_scope_links += 1
            local_scope = local_project.get("scopesById", {}).get(scope_id)
            if not local_scope:
                continue
            if local_scope.get("groupTaskSetId") != group_task_set_id:
                local_scope["groupTaskSetId"] = group_task_set_id
                updated_scope_links += 1

    return {
        "enabled": True,
        "tenantId": tenant_id,
        "projectCount": len(project_ids),
        "liveProjectCount": len(live_projects),
        "missingProjectIds": missing_projects,
        "liveScopeLinkCount": live_scope_links,
        "updatedScopeLinkCount": updated_scope_links,
    }


async def _post_logic_app(
    client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    try:
        response = await client.post(endpoint, json=payload, headers=headers)
        body_text = response.text
        try:
            body_json = response.json()
        except ValueError:
            body_json = None
        return {
            "candidateId": payload.get("OrchestrationRunId"),
            "startedAt": started,
            "completedAt": datetime.now(UTC).isoformat(),
            "httpStatus": response.status_code,
            "ok": 200 <= response.status_code < 300,
            "json": body_json,
            "text": body_text[:2000],
        }
    except Exception as exc:
        return {
            "candidateId": payload.get("OrchestrationRunId"),
            "startedAt": started,
            "completedAt": datetime.now(UTC).isoformat(),
            "httpStatus": None,
            "ok": False,
            "errorType": type(exc).__name__,
            "error": str(exc),
        }


async def _post_add_group_task_direct(
    client: httpx.AsyncClient,
    *,
    base_url: str | None,
    function_key: str | None,
    candidate: dict[str, Any],
    max_retries: int,
    base_delay_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    if not base_url or not function_key:
        return {
            "candidateId": candidate["candidateId"],
            "startedAt": started,
            "completedAt": datetime.now(UTC).isoformat(),
            "httpStatus": None,
            "ok": False,
            "writeMode": "direct-task-api",
            "error": "TASK_FUNCTION_BASE_URL and TASK_FUNCTION_KEY are required.",
        }
    url = (
        f"{base_url.rstrip('/')}/api/addgrouptasktogts/"
        f"{candidate['groupTaskSetIdBeforeWrite']}/{candidate['tenantId']}/"
    )
    group_task = (
        (candidate.get("logicAppPayload") or {}).get("grouptask")
        or (candidate.get("logicAppPayload") or {}).get("GroupTask")
        or []
    )[0]
    attempts: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, params={"code": function_key}, json=group_task)
            last_result = {
                "candidateId": candidate["candidateId"],
                "startedAt": started,
                "completedAt": datetime.now(UTC).isoformat(),
                "httpStatus": response.status_code,
                "ok": 200 <= response.status_code < 300,
                "writeMode": "direct-task-api",
                "text": response.text[:2000],
            }
        except Exception as exc:
            last_result = {
                "candidateId": candidate["candidateId"],
                "startedAt": started,
                "completedAt": datetime.now(UTC).isoformat(),
                "httpStatus": None,
                "ok": False,
                "writeMode": "direct-task-api",
                "errorType": type(exc).__name__,
                "error": str(exc),
            }
        attempts.append(
            {
                "attempt": attempt + 1,
                "httpStatus": last_result.get("httpStatus"),
                "ok": last_result.get("ok"),
                "errorType": last_result.get("errorType"),
            }
        )
        if last_result.get("ok") or not _is_retryable_write_response(last_result):
            last_result["attempts"] = attempts
            return last_result
        if attempt < max_retries:
            await asyncio.sleep(base_delay_seconds * (2 ** attempt))
    assert last_result is not None
    last_result["attempts"] = attempts
    return last_result


def _cosmos_group_task_set_container(connection_string: str):
    client = CosmosClient.from_connection_string(connection_string)
    return client.get_database_client("bloomskyHealth").get_container_client("GroupTaskSet")


async def _append_group_task_cosmos(
    container,
    *,
    candidate: dict[str, Any],
    max_retries: int,
    base_delay_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    group_task = (
        (candidate.get("logicAppPayload") or {}).get("grouptask")
        or (candidate.get("logicAppPayload") or {}).get("GroupTask")
        or []
    )[0]
    attempts: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.to_thread(
                container.patch_item,
                item=candidate["groupTaskSetIdBeforeWrite"],
                partition_key=candidate["tenantId"],
                patch_operations=[
                    {"op": "add", "path": "/GroupTask/-", "value": group_task}
                ],
            )
            last_result = {
                "candidateId": candidate["candidateId"],
                "startedAt": started,
                "completedAt": datetime.now(UTC).isoformat(),
                "httpStatus": 200,
                "ok": True,
                "writeMode": "direct-cosmos",
                "etag": response.get("_etag") if isinstance(response, dict) else None,
            }
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            last_result = {
                "candidateId": candidate["candidateId"],
                "startedAt": started,
                "completedAt": datetime.now(UTC).isoformat(),
                "httpStatus": status_code,
                "ok": False,
                "writeMode": "direct-cosmos",
                "errorType": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        attempts.append(
            {
                "attempt": attempt + 1,
                "httpStatus": last_result.get("httpStatus"),
                "ok": last_result.get("ok"),
                "errorType": last_result.get("errorType"),
            }
        )
        if last_result.get("ok") or not _is_retryable_write_response(last_result):
            last_result["attempts"] = attempts
            return last_result
        if attempt < max_retries:
            await asyncio.sleep(base_delay_seconds * (2 ** attempt))
    assert last_result is not None
    last_result["attempts"] = attempts
    return last_result


async def _post_logic_app_with_retries(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    max_retries: int,
    base_delay_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        response = await _post_logic_app(client, endpoint, payload, headers)
        attempts.append(
            {
                "attempt": attempt + 1,
                "httpStatus": response.get("httpStatus"),
                "ok": response.get("ok"),
                "errorType": response.get("errorType"),
            }
        )
        if response.get("ok") or not _is_retryable_write_response(response):
            response["attempts"] = attempts
            return response
        if attempt < max_retries:
            await asyncio.sleep(base_delay_seconds * (2 ** attempt))
    attempts[-1]["final"] = True
    response["attempts"] = attempts
    return response


def _is_retryable_write_response(response: dict[str, Any]) -> bool:
    status = response.get("httpStatus")
    if status is None:
        return True
    if status in {429, 500, 502, 503, 504}:
        return True
    try:
        numeric_status = int(status)
    except (TypeError, ValueError):
        return False
    return numeric_status in {429, 500, 502, 503, 504}


async def _fetch_group_task_set(
    client: httpx.AsyncClient,
    *,
    base_url: str | None,
    function_key: str | None,
    group_task_set_id: str | None,
    tenant_id: str,
) -> dict[str, Any] | None:
    if not base_url or not function_key or not group_task_set_id:
        return None
    if group_task_set_id == ZERO_GUID:
        return None
    url = (
        f"{base_url.rstrip('/')}/api/grouptaskset/{group_task_set_id}/{tenant_id}"
        f"?code={function_key}"
    )
    try:
        response = await client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"verificationFetchError": type(exc).__name__, "error": str(exc)}


async def _fetch_group_task_set_for_mode(
    client: httpx.AsyncClient,
    *,
    cosmos_container,
    base_url: str | None,
    function_key: str | None,
    group_task_set_id: str | None,
    tenant_id: str,
) -> dict[str, Any] | None:
    if cosmos_container is not None:
        return await _fetch_group_task_set_cosmos(
            cosmos_container,
            group_task_set_id=group_task_set_id,
            tenant_id=tenant_id,
        )
    return await _fetch_group_task_set(
        client,
        base_url=base_url,
        function_key=function_key,
        group_task_set_id=group_task_set_id,
        tenant_id=tenant_id,
    )


async def _fetch_group_task_set_cosmos(
    container,
    *,
    group_task_set_id: str | None,
    tenant_id: str,
) -> dict[str, Any] | None:
    if not group_task_set_id or group_task_set_id == ZERO_GUID:
        return None
    try:
        return await asyncio.to_thread(
            container.read_item,
            item=group_task_set_id,
            partition_key=tenant_id,
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            return None
        return {"verificationFetchError": type(exc).__name__, "error": str(exc)}


def _verify_write(
    *,
    candidate: dict[str, Any],
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    write_response: dict[str, Any],
    group_task_set_id: str | None,
) -> dict[str, Any]:
    marker = f"candidateId={candidate['candidateId']}"
    before_ids = _group_task_ids(before)
    after_tasks = (after or {}).get("GroupTask") or (after or {}).get("grouptask") or []
    new_tasks = [
        task
        for task in after_tasks
        if task.get("GroupTaskID") not in before_ids
        or marker in str(task.get("GroupTaskNotes") or task.get("groupetasknotes") or "")
    ]
    matching = [
        task
        for task in after_tasks
        if marker in str(task.get("GroupTaskNotes") or task.get("groupetasknotes") or "")
    ]
    created = matching or new_tasks
    task = created[-1] if created else None
    return {
        "candidateId": candidate["candidateId"],
        "sourceLine": candidate["sourceLine"],
        "tenantId": candidate["tenantId"],
        "projectId": candidate["projectId"],
        "scopeId": candidate["scopeId"],
        "groupTaskSetId": group_task_set_id,
        "logicAppOk": bool(write_response.get("ok")),
        "verificationOk": bool(task)
        and (after or {}).get("TenantID") == candidate["tenantId"]
        and (after or {}).get("ProjectID") == candidate["projectId"],
        "beforeGroupTaskCount": len((before or {}).get("GroupTask") or []),
        "afterGroupTaskCount": len(after_tasks),
        "createdGroupTask": task,
        "fetchError": (after or {}).get("verificationFetchError") if isinstance(after, dict) else None,
    }


def _find_duplicate_task(
    group_task_set: dict[str, Any] | None, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    if not group_task_set:
        return None
    idempotency_key = candidate.get("headers", {}).get("x-taslow-idempotency-key", "")
    candidate_marker = f"candidateId={candidate['candidateId']}"
    idempotency_marker = f"idempotencyKey={idempotency_key}"
    for task in (group_task_set.get("GroupTask") or group_task_set.get("grouptask") or []):
        notes = str(task.get("GroupTaskNotes") or task.get("groupetasknotes") or "")
        if candidate_marker in notes or (idempotency_key and idempotency_marker in notes):
            return task
    return None


def _duplicate_skip_response(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidateId": candidate["candidateId"],
        "startedAt": datetime.now(UTC).isoformat(),
        "completedAt": datetime.now(UTC).isoformat(),
        "httpStatus": "skipped_duplicate",
        "ok": True,
        "skipped": True,
        "reason": "existing_task_matches_idempotency_key_or_candidate",
    }


def _verify_duplicate_skip(
    *,
    candidate: dict[str, Any],
    group_task_set: dict[str, Any] | None,
    write_response: dict[str, Any],
    duplicate_task: dict[str, Any],
) -> dict[str, Any]:
    tasks = (group_task_set or {}).get("GroupTask") or (group_task_set or {}).get("grouptask") or []
    return {
        "candidateId": candidate["candidateId"],
        "sourceLine": candidate["sourceLine"],
        "tenantId": candidate["tenantId"],
        "projectId": candidate["projectId"],
        "scopeId": candidate["scopeId"],
        "groupTaskSetId": candidate["groupTaskSetIdBeforeWrite"],
        "logicAppOk": bool(write_response.get("ok")),
        "verificationOk": bool(duplicate_task)
        and (group_task_set or {}).get("TenantID") == candidate["tenantId"]
        and (group_task_set or {}).get("ProjectID") == candidate["projectId"],
        "duplicateSkip": True,
        "beforeGroupTaskCount": len(tasks),
        "afterGroupTaskCount": len(tasks),
        "createdGroupTask": duplicate_task,
        "fetchError": (group_task_set or {}).get("verificationFetchError")
        if isinstance(group_task_set, dict)
        else None,
    }


def _group_task_ids(group_task_set: dict[str, Any] | None) -> set[str]:
    tasks = (group_task_set or {}).get("GroupTask") or (group_task_set or {}).get("grouptask") or []
    return {str(task.get("GroupTaskID")) for task in tasks if task.get("GroupTaskID")}


def _load_jsonl_by_source_line(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_line = int(row.get("sourceLine") or line_no)
            rows[source_line] = row
    return rows


def _load_project_context(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    projects_by_id: dict[str, dict[str, Any]] = {}
    for project in payload.get("projects", []):
        project_id = _first(project, "projectId", "ProjectID", "id")
        scopes_by_id = {}
        for scope in project.get("projectScopes") or project.get("scopes") or project.get("ProjectScopes") or []:
            scope_id = _first(scope, "scopeId", "ScopeID")
            if not scope_id:
                continue
            scopes_by_id[scope_id] = {
                "scopeId": scope_id,
                "scopeTitle": _first(scope, "scopeTitle", "ProjectScopeAreaTitle", "title") or "",
                "scopeDescription": _first(scope, "scopeDescription", "ProjectScopeArea", "description")
                or "",
                "groupTaskSetId": _first(scope, "groupTaskSetId", "GroupTaskSetID") or "",
            }
        projects_by_id[project_id] = {
            "projectId": project_id,
            "projectName": _first(project, "projectName", "ProjectName") or "",
            "scopesById": scopes_by_id,
        }
    return {"tenantId": payload.get("tenantId"), "projectsById": projects_by_id}


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _idempotency_key(tenant_id: str, request: dict[str, Any], task: dict[str, Any]) -> str:
    message_id = request.get("internetMessageId") or request.get("messageId")
    return "|".join(
        [
            tenant_id,
            str(message_id),
            str(request.get("direction") or ""),
            str(task.get("sourceTaskId") or ""),
            str(task.get("assigneeEmail") or "").lower(),
        ]
    )


def _stable_guid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_SYSTEM}:{value.lower()}"))


def _parse_source_lines(value: str) -> set[int]:
    if not value:
        return set()
    return {
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    }


def _load_excluded_source_lines(stage2_run_dirs: list[Path]) -> set[int]:
    source_lines: set[int] = set()
    for run_dir in stage2_run_dirs:
        verification_path = run_dir / "stage2_task_write_verification.jsonl"
        if not verification_path.exists():
            continue
        with verification_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("verificationOk") and row.get("sourceLine"):
                    source_lines.add(int(row["sourceLine"]))
    return source_lines


def _truncate(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    return cleaned[:limit]


def _summary(
    args: argparse.Namespace,
    run_id: str,
    candidates: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
) -> dict[str, Any]:
    response_counts = Counter(str(row.get("httpStatus")) for row in responses)
    candidates_by_id = {row["candidateId"]: row for row in candidates}
    return {
        "stage2RunId": run_id,
        "syntheticRunId": args.synthetic_run_id,
        "tenantId": args.tenant_id,
        "dryRun": args.dry_run,
        "allowCreateGts": args.allow_create_gts,
        "existingGtsWriteMode": args.existing_gts_write_mode,
        "liveProjectContextRefresh": not args.skip_live_project_context_refresh,
        "excludedSourceLineCount": len(args.exclude_source_lines),
        "candidateCount": len(candidates),
        "writeAttemptCount": len(responses),
        "logicAppPostAttemptCount": sum(len(row.get("attempts") or [row]) for row in responses),
        "retriedWriteCount": sum(1 for row in responses if len(row.get("attempts") or []) > 1),
        "writeSuccessCount": sum(1 for row in responses if row.get("ok")),
        "duplicateSkipCount": sum(1 for row in responses if row.get("skipped")),
        "verificationSuccessCount": sum(1 for row in verifications if row.get("verificationOk")),
        "existingGroupTaskSetCandidateCount": sum(
            1 for row in candidates if row.get("usesExistingGroupTaskSet")
        ),
        "createGroupTaskSetCandidateCount": sum(
            1 for row in candidates if not row.get("usesExistingGroupTaskSet")
        ),
        "httpStatusCounts": dict(response_counts),
        "sourceLines": [row["sourceLine"] for row in candidates],
        "projectIds": sorted({row["projectId"] for row in candidates}),
        "scopeIds": sorted({row["scopeId"] for row in candidates}),
        "verifiedGroupTaskSetIds": [
            row.get("groupTaskSetId") for row in verifications if row.get("groupTaskSetId")
        ],
        "createdGroupTaskSetIds": [
            row.get("groupTaskSetId")
            for row in verifications
            if row.get("groupTaskSetId")
            and not row.get("duplicateSkip")
            and (
                candidates_by_id.get(row.get("candidateId"), {}).get("groupTaskSetIdBeforeWrite")
                == ZERO_GUID
            )
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Synthetic Task Write Smoke - {summary['stage2RunId']}",
        "",
        f"- Tenant: {summary['tenantId']}",
        f"- Dry run: {summary['dryRun']}",
        f"- Existing GTS write mode: {summary['existingGtsWriteMode']}",
        f"- Excluded source lines: {summary['excludedSourceLineCount']}",
        f"- Candidate count: {summary['candidateCount']}",
        f"- Existing GroupTaskSet candidates: {summary['existingGroupTaskSetCandidateCount']}",
        f"- Create GroupTaskSet candidates: {summary['createGroupTaskSetCandidateCount']}",
        f"- Write attempts: {summary['writeAttemptCount']}",
        f"- Write successes: {summary['writeSuccessCount']}",
        f"- Duplicate skips: {summary['duplicateSkipCount']}",
        f"- Verification successes: {summary['verificationSuccessCount']}",
        f"- Source lines: {summary['sourceLines']}",
        f"- HTTP statuses: {summary['httpStatusCounts']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
