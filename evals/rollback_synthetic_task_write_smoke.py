from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


SOURCE_SYSTEM = "taslow-ai-synthetic-evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback synthetic task-write smoke runs from Taslow Dev."
    )
    parser.add_argument("--stage2-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--synthetic-run-id")
    parser.add_argument("--task-function-base-url", default=os.getenv("TASK_FUNCTION_BASE_URL"))
    parser.add_argument("--task-function-key", default=os.getenv("TASK_FUNCTION_KEY"))
    parser.add_argument("--project-callback-base-url", default=os.getenv("PROJECT_SCOPE_LINK_CALLBACK_BASE_URL"))
    parser.add_argument(
        "--project-callback-function-key",
        default=os.getenv("PROJECT_SCOPE_LINK_CALLBACK_FUNCTION_KEY"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-created-gts", action="store_true")
    parser.add_argument("--skip-clear-project-scope-links", action="store_true")
    parser.add_argument("--confirm-synthetic-run-id")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    args = parser.parse_args()

    if not args.dry_run and not args.delete_created_gts:
        raise SystemExit("Refusing to run: pass --dry-run or --delete-created-gts explicitly.")
    if args.delete_created_gts and not args.confirm_synthetic_run_id:
        raise SystemExit("--confirm-synthetic-run-id is required for delete rollback.")
    if args.delete_created_gts and not args.task_function_base_url:
        raise SystemExit("TASK_FUNCTION_BASE_URL or --task-function-base-url is required.")
    if args.delete_created_gts and not args.task_function_key:
        raise SystemExit("TASK_FUNCTION_KEY or --task-function-key is required.")
    if args.delete_created_gts and not args.skip_clear_project_scope_links:
        if not args.project_callback_base_url or not args.project_callback_function_key:
            raise SystemExit(
                "Clearing Project scope links requires PROJECT_SCOPE_LINK_CALLBACK_BASE_URL "
                "and PROJECT_SCOPE_LINK_CALLBACK_FUNCTION_KEY."
            )

    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-rollback"
    out_dir = args.stage2_run_dir[0].parent / "rollback" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_stage2_rows(args.stage2_run_dir, args.synthetic_run_id)
    if args.confirm_synthetic_run_id:
        mismatches = [
            row for row in rows if row.get("syntheticRunId") != args.confirm_synthetic_run_id
        ]
        if mismatches:
            raise SystemExit(
                "--confirm-synthetic-run-id does not match every selected Stage 2 row."
            )

    plan: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        for row in rows:
            group_task_set = await _fetch_group_task_set(
                client,
                base_url=args.task_function_base_url,
                function_key=args.task_function_key,
                group_task_set_id=row["groupTaskSetId"],
                tenant_id=args.tenant_id,
            )
            plan_row = _plan_row(row, group_task_set, args.tenant_id)
            plan.append(plan_row)

            if args.dry_run:
                continue

            if not plan_row["safeToDeleteGroupTaskSet"]:
                results.append(
                    {
                        **_identity(row),
                        "action": "skip",
                        "ok": False,
                        "reason": "; ".join(plan_row["safetyReasons"]),
                    }
                )
                continue

            delete_result = await _delete_group_task_set(
                client,
                base_url=args.task_function_base_url,
                function_key=args.task_function_key,
                group_task_set_id=row["groupTaskSetId"],
                tenant_id=args.tenant_id,
            )
            link_result: dict[str, Any] | None = None
            if delete_result.get("ok") and not args.skip_clear_project_scope_links:
                link_result = await _clear_project_scope_link(
                    client,
                    base_url=args.project_callback_base_url,
                    secret=args.project_callback_function_key,
                    row=row,
                    tenant_id=args.tenant_id,
                )
            verify_after = await _fetch_group_task_set(
                client,
                base_url=args.task_function_base_url,
                function_key=args.task_function_key,
                group_task_set_id=row["groupTaskSetId"],
                tenant_id=args.tenant_id,
            )
            results.append(
                {
                    **_identity(row),
                    "action": "delete_created_gts",
                    "ok": bool(delete_result.get("ok"))
                    and (link_result is None or bool(link_result.get("ok")))
                    and verify_after is None,
                    "delete": delete_result,
                    "clearProjectScopeLink": link_result,
                    "verification": {
                        "groupTaskSetDeleted": verify_after is None,
                    },
                }
            )

    summary = _summary(args, rows, plan, results)
    _write_jsonl(out_dir / "stage2_rollback_plan.jsonl", plan)
    _write_jsonl(out_dir / "stage2_rollback_results.jsonl", results)
    _write_json(out_dir / "stage2_rollback_summary.json", summary)
    _write_summary_md(out_dir / "stage2_rollback_summary.md", summary)
    print(json.dumps({"outputDir": str(out_dir), **summary}, indent=2))


def _load_stage2_rows(stage2_run_dirs: list[Path], synthetic_run_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage2_run_dir in stage2_run_dirs:
        summary = _load_json(stage2_run_dir / "stage2_task_write_summary.json")
        run_synthetic_id = summary.get("syntheticRunId")
        if synthetic_run_id and run_synthetic_id != synthetic_run_id:
            continue
        candidates = {
            row["candidateId"]: row
            for row in _load_jsonl(stage2_run_dir / "stage2_write_candidates.jsonl")
        }
        for verification in _load_jsonl(stage2_run_dir / "stage2_task_write_verification.jsonl"):
            if not verification.get("verificationOk"):
                continue
            candidate = candidates.get(verification.get("candidateId")) or {}
            rows.append(
                {
                    "stage2RunDir": str(stage2_run_dir),
                    "stage2RunId": summary.get("stage2RunId"),
                    "syntheticRunId": run_synthetic_id,
                    "candidateId": verification.get("candidateId"),
                    "sourceLine": verification.get("sourceLine"),
                    "tenantId": verification.get("tenantId"),
                    "projectId": verification.get("projectId"),
                    "scopeId": verification.get("scopeId"),
                    "groupTaskSetId": verification.get("groupTaskSetId"),
                    "taskTitle": (verification.get("createdGroupTask") or {}).get("GroupTaskTitle"),
                    "assigneeEmail": candidate.get("assigneeEmail"),
                }
            )
    return rows


def _plan_row(
    row: dict[str, Any], group_task_set: dict[str, Any] | None, expected_tenant_id: str
) -> dict[str, Any]:
    reasons: list[str] = []
    if not group_task_set:
        reasons.append("group_task_set_not_found")
    else:
        tenant_id = _first(group_task_set, "TenantID", "tenantId", "tenantid")
        project_id = _first(group_task_set, "ProjectID", "projectId", "caseid")
        if tenant_id != expected_tenant_id:
            reasons.append("tenant_mismatch")
        if project_id and project_id != row.get("projectId"):
            reasons.append("project_mismatch")
        tasks = _tasks(group_task_set)
        if not tasks:
            reasons.append("no_group_tasks")
        if not _all_tasks_match_run(tasks, row.get("syntheticRunId", "")):
            reasons.append("non_synthetic_or_different_run_tasks_present")

    return {
        **_identity(row),
        "taskTitle": row.get("taskTitle"),
        "assigneeEmail": row.get("assigneeEmail"),
        "safeToDeleteGroupTaskSet": len(reasons) == 0,
        "safetyReasons": reasons,
        "groupTaskCount": len(_tasks(group_task_set or {})),
        "plannedAction": "delete_created_gts_and_clear_scope_link",
    }


def _all_tasks_match_run(tasks: list[dict[str, Any]], synthetic_run_id: str) -> bool:
    if not synthetic_run_id:
        return False
    for task in tasks:
        notes = str(task.get("GroupTaskNotes") or task.get("groupTaskNotes") or "")
        if f"sourceSystem={SOURCE_SYSTEM}" not in notes:
            return False
        if f"syntheticRunId={synthetic_run_id}" not in notes:
            return False
    return True


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
    url = f"{base_url.rstrip()}/api/grouptaskset/{group_task_set_id}/{tenant_id}"
    response = await client.get(url, params={"code": function_key})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def _delete_group_task_set(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    function_key: str,
    group_task_set_id: str,
    tenant_id: str,
) -> dict[str, Any]:
    url = f"{base_url.rstrip()}/api/grouptaskset/{group_task_set_id}/{tenant_id}"
    response = await client.delete(url, params={"code": function_key})
    return {
        "httpStatus": response.status_code,
        "ok": 200 <= response.status_code < 300 or response.status_code == 404,
        "text": response.text[:1000],
    }


async def _clear_project_scope_link(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    row: dict[str, Any],
    tenant_id: str,
) -> dict[str, Any]:
    url = (
        f"{base_url.rstrip()}/projects/"
        f"{tenant_id}/{row['projectId']}/scopes/link-gts"
    )
    payload = {
        "tenantId": tenant_id,
        "projectId": row["projectId"],
        "mappings": [
            {
                "scopeId": row["scopeId"],
                "groupTaskSetId": "",
                "orchestrationRunId": f"rollback-{row['candidateId']}",
            }
        ],
    }
    response = await client.patch(
        url,
        headers={"Content-Type": "application/json", "x-scope-sync-secret": secret},
        json=payload,
    )
    return {
        "httpStatus": response.status_code,
        "ok": 200 <= response.status_code < 300,
        "text": response.text[:1000],
    }


def _summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rollbackMode": "dry-run" if args.dry_run else "delete-created-gts",
        "tenantId": args.tenant_id,
        "inputStage2RunDirs": [str(path) for path in args.stage2_run_dir],
        "candidateCount": len(rows),
        "safeToDeleteGroupTaskSetCount": sum(
            1 for row in plan if row.get("safeToDeleteGroupTaskSet")
        ),
        "unsafeCount": sum(1 for row in plan if not row.get("safeToDeleteGroupTaskSet")),
        "resultCount": len(results),
        "successCount": sum(1 for row in results if row.get("ok")),
        "resultActions": dict(Counter(str(row.get("action")) for row in results)),
        "resultStatuses": dict(
            Counter(str((row.get("delete") or {}).get("httpStatus")) for row in results)
        ),
    }


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "syntheticRunId": row.get("syntheticRunId"),
        "candidateId": row.get("candidateId"),
        "sourceLine": row.get("sourceLine"),
        "tenantId": row.get("tenantId"),
        "projectId": row.get("projectId"),
        "scopeId": row.get("scopeId"),
        "groupTaskSetId": row.get("groupTaskSetId"),
    }


def _tasks(group_task_set: dict[str, Any]) -> list[dict[str, Any]]:
    return group_task_set.get("GroupTask") or group_task_set.get("grouptask") or []


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Synthetic Task Rollback - {datetime.now(UTC).isoformat()}",
        "",
        f"- Mode: {summary['rollbackMode']}",
        f"- Tenant: {summary['tenantId']}",
        f"- Candidates: {summary['candidateCount']}",
        f"- Safe to delete: {summary['safeToDeleteGroupTaskSetCount']}",
        f"- Unsafe: {summary['unsafeCount']}",
        f"- Results: {summary['resultCount']}",
        f"- Successes: {summary['successCount']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
