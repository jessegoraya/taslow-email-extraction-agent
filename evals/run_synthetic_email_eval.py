from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taslow_email_extraction_agent.clients.project_client import InMemoryProjectClient
from taslow_email_extraction_agent.clients.project_search_client import AzureProjectSearchClient
from taslow_email_extraction_agent.clients.task_history_client import InMemoryTaskHistoryClient
from taslow_email_extraction_agent.config import Settings
from taslow_email_extraction_agent.executors.task_detection import FoundryTaskExtractor
from taslow_email_extraction_agent.models import (
    AssociatedPerson,
    EmailExtractionRequest,
    ProjectContext,
    ProjectScope,
)
from taslow_email_extraction_agent.services import WorkflowServices
from taslow_email_extraction_agent.workflow import run_email_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Taslow synthetic email evaluation.")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--project-context", type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--source-lines", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-label", default="local-eval")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    asyncio.run(run_eval(args))


async def run_eval(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    run_id = datetime.now().strftime(f"%Y%m%d-%H%M%S-{args.run_label}")
    run_dir = args.results_root / run_id
    failures_dir = run_dir / "failures"
    failures_dir.mkdir(parents=True)

    selected, selected_lines = _select_requests(
        args.requests,
        args.sample_size,
        args.seed,
        args.source_lines,
    )
    _write_jsonl(run_dir / "selected_requests.jsonl", selected)

    settings = Settings()
    projects = _load_projects(args.project_context) if args.project_context else []
    services = WorkflowServices(
        settings=settings,
        task_extractor=FoundryTaskExtractor(settings),
        project_client=InMemoryProjectClient(projects),
        task_history_client=InMemoryTaskHistoryClient(),
        project_search_client=AzureProjectSearchClient(settings)
        if settings.project_search_provider in {"azure-ai-search", "shadow"}
        else None,
    )

    manifest = {
        "evaluationRunId": run_id,
        "startedAt": datetime.now(UTC).isoformat(),
        "requestPath": str(args.requests),
        "answerKeyPath": str(args.answer_key),
        "answerKeyLoadedAfterRawExecution": None,
        "projectContextPath": str(args.project_context) if args.project_context else None,
        "sampleSize": args.sample_size,
        "selectedSourceLines": selected_lines,
        "projectCountLoaded": len(projects),
        "modelDeploymentName": settings.azure_ai_model_deployment_name,
        "projectSearchProvider": settings.project_search_provider,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    raw_rows = await _execute_requests(selected, services, run_dir)

    answer_by_key = _load_answer_key(args.answer_key)
    manifest["answerKeyLoadedAfterRawExecution"] = True
    manifest["rawExecutionCompletedAt"] = datetime.now(UTC).isoformat()
    _write_json(run_dir / "run_manifest.json", manifest)

    summary, details, failure_buckets = _score_rows(
        raw_rows,
        answer_by_key,
        selected_lines,
        run_id,
        args.answer_key,
    )
    _write_json(run_dir / "scoring_summary.json", summary)
    _write_jsonl(run_dir / "scoring_details.jsonl", details)
    _write_json(run_dir / "cost_usage_summary.json", _usage_summary(raw_rows))
    for failure, rows in failure_buckets.items():
        _write_jsonl(failures_dir / f"{failure}.jsonl", rows)
    _write_jsonl(failures_dir / "all_failures.jsonl", [row for row in details if not row["passed"]])
    _write_summary_md(run_dir / "scoring_summary.md", summary)

    print(json.dumps({"runDir": str(run_dir), **summary}, indent=2))


async def _execute_requests(
    selected: list[dict[str, Any]],
    services: WorkflowServices,
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "agent_responses.jsonl").open("w", encoding="utf-8") as handle:
        for index, request_row in enumerate(selected, start=1):
            request = EmailExtractionRequest.model_validate(request_row)
            try:
                response = await run_email_extraction(request, services)
                response_json = response.to_jsonable()
            except Exception as exc:
                response_json = {
                    "status": "failed",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "tenantId": request_row.get("tenantId"),
                    "graphEventId": request_row.get("graphEventId"),
                    "messageId": request_row.get("messageId"),
                    "tasks": [],
                }
            row = {
                "runIndex": index,
                "sourceLine": request_row["sourceLine"],
                "graphEventId": request_row.get("graphEventId"),
                "messageId": request_row.get("messageId"),
                "internetMessageId": request_row.get("internetMessageId"),
                "response": response_json,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    return rows


def _score_rows(
    raw_rows: list[dict[str, Any]],
    answer_by_key: dict[str, dict[str, Any]],
    selected_lines: list[int],
    run_id: str,
    answer_key_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    details: list[dict[str, Any]] = []
    by_scenario: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0}
    )
    by_sub: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    failure_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counter: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()

    for raw in raw_rows:
        response = raw.get("response") or {}
        status_counter[_normalize_status(response.get("status")) or "unknown"] += 1
        diagnostics = response.get("diagnostics") or {}
        fallback_counter[str(diagnostics.get("modelFallbackUsed", False)).lower()] += 1

        answer = _find_answer(answer_by_key, raw)
        if not answer:
            score = {"passed": False, "failures": ["missing_answer_key"]}
            scenario = "UNKNOWN"
            sub = None
        else:
            score = _score_one(raw, answer)
            scenario = answer.get("scenarioId") or "UNKNOWN"
            sub = answer.get("subScenarioId")
        detail = {
            "sourceLine": raw["sourceLine"],
            "graphEventId": raw.get("graphEventId"),
            "messageId": raw.get("messageId"),
            "scenarioId": scenario,
            "subScenarioId": sub,
            **score,
            "diagnostics": diagnostics,
        }
        details.append(detail)
        by_scenario[scenario]["total"] += 1
        by_scenario[scenario]["passed" if score["passed"] else "failed"] += 1
        if sub:
            by_sub[sub]["total"] += 1
            by_sub[sub]["passed" if score["passed"] else "failed"] += 1
        for failure in score["failures"]:
            failure_buckets[failure].append(detail)

    passed = sum(1 for detail in details if detail["passed"])
    for bucket in [*by_scenario.values(), *by_sub.values()]:
        bucket["passRate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0

    summary = {
        "evaluationRunId": run_id,
        "scoredAt": datetime.now(UTC).isoformat(),
        "answerKeyPath": str(answer_key_path),
        "answerKeyLoadedAfterRawExecution": True,
        "total": len(details),
        "passed": passed,
        "failed": len(details) - passed,
        "passRate": round(passed / len(details), 4) if details else 0,
        "selectedSourceLines": selected_lines,
        "actualStatuses": dict(status_counter),
        "modelFallbackUsed": dict(fallback_counter),
        "byScenario": dict(sorted(by_scenario.items())),
        "bySubScenario": dict(sorted(by_sub.items())),
        "failureCounts": {key: len(value) for key, value in sorted(failure_buckets.items())},
    }
    return summary, details, failure_buckets


def _score_one(raw_row: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    response = raw_row["response"]
    actual_status = _normalize_status(response.get("status"))
    expected_status = _normalize_status(answer.get("expectedStatus"))
    actual_tasks = response.get("tasks") or []
    expected_tasks = answer.get("expectedTasks") or []
    expected_count = answer.get("expectedTaskCount")
    if expected_count is None:
        expected_count = len(expected_tasks)
    project_match = response.get("projectMatch") or {}
    actual_scopes = {task.get("scopeId") for task in actual_tasks if task.get("scopeId")}
    actual_assignees = {
        str(task.get("assigneeEmail", "")).lower()
        for task in actual_tasks
        if task.get("assigneeEmail")
    }
    actual_due_dates = {task.get("dueDate") for task in actual_tasks if task.get("dueDate")}
    expected_assignees = {
        str(
            task.get("assigneeEmail")
            or task.get("expectedAssigneeEmail")
            or task.get("assignee")
        ).lower()
        for task in expected_tasks
        if task.get("assigneeEmail") or task.get("expectedAssigneeEmail") or task.get("assignee")
    }
    expected_due_present = any(
        task.get("dueDate") or task.get("expectedDueDate") for task in expected_tasks
    )

    failures: list[str] = []
    if expected_status and actual_status != expected_status:
        failures.append("status")
    if len(actual_tasks) != int(expected_count or 0):
        failures.append("task_count")
    if (
        answer.get("expectedProjectId")
        and expected_status == "tasks_ready"
        and project_match.get("projectId") != answer.get("expectedProjectId")
    ):
        failures.append("project")
    if (
        answer.get("expectedScopeId")
        and expected_status == "tasks_ready"
        and answer.get("expectedScopeId") not in actual_scopes
    ):
        failures.append("scope")
    if (
        expected_assignees
        and expected_status == "tasks_ready"
        and not expected_assignees.issubset(actual_assignees)
    ):
        failures.append("assignee")
    if expected_due_present and expected_status == "tasks_ready" and not actual_due_dates:
        failures.append("due_date")

    return {
        "passed": not failures,
        "failures": failures,
        "actualStatus": actual_status,
        "expectedStatus": expected_status,
        "actualTaskCount": len(actual_tasks),
        "expectedTaskCount": expected_count,
        "actualProjectId": project_match.get("projectId"),
        "expectedProjectId": answer.get("expectedProjectId"),
    }


def _usage_summary(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in raw_rows:
        diagnostics = (row.get("response") or {}).get("diagnostics") or {}
        totals["modelInputTokenCount"] += diagnostics.get("modelInputTokenCount") or 0
        totals["modelOutputTokenCount"] += diagnostics.get("modelOutputTokenCount") or 0
        totals["searchQueryCount"] += diagnostics.get("searchQueryCount") or 0
        totals["scopeSearchQueryCount"] += diagnostics.get("scopeSearchQueryCount") or 0
    return {"totals": dict(totals), "recordCount": len(raw_rows)}


def _select_requests(
    requests_path: Path,
    sample_size: int,
    seed: int,
    source_lines_path: Path | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    all_rows = []
    with requests_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            row["sourceLine"] = line_no
            all_rows.append(row)
    if source_lines_path:
        selected_lines = json.loads(source_lines_path.read_text(encoding="utf-8-sig"))
        if isinstance(selected_lines, dict):
            selected_lines = selected_lines["selectedSourceLines"]
        by_line = {row["sourceLine"]: row for row in all_rows}
        return [by_line[line] for line in selected_lines], selected_lines
    rng = random.Random(seed)
    selected = rng.sample(all_rows, sample_size)
    selected.sort(key=lambda row: row["sourceLine"])
    return selected, [row["sourceLine"] for row in selected]


def _load_answer_key(answer_key_path: Path) -> dict[str, dict[str, Any]]:
    answer_by_key = {}
    with answer_key_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in _email_keys(row):
                answer_by_key[key] = row
    return answer_by_key


def _find_answer(
    answer_by_key: dict[str, dict[str, Any]],
    request_row: dict[str, Any],
) -> dict[str, Any] | None:
    for key in _email_keys(request_row):
        if key in answer_by_key:
            return answer_by_key[key]
    return None


def _email_keys(row: dict[str, Any]) -> set[str]:
    keys = set()
    for field in [
        "emailId",
        "graphEventId",
        "messageId",
        "internetMessageId",
        "idempotencyKey",
        "correlationId",
    ]:
        value = row.get(field)
        if not value:
            continue
        keys.add(str(value))
        match = re.search(r"(\d{3,6})", str(value))
        if match:
            number = int(match.group(1))
            keys.update(
                {
                    str(number),
                    f"{number:06d}",
                    f"synthetic-email-{number:06d}",
                    f"synthetic-email-{number:03d}",
                    f"email-{number:06d}",
                }
            )
    if row.get("sourceLine"):
        number = int(row["sourceLine"])
        keys.update({str(number), f"{number:06d}", f"synthetic-email-{number:06d}"})
    return keys


def _load_projects(path: Path) -> list[ProjectContext]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    projects = []
    for tenant in payload.get("tenants", []):
        for row in tenant.get("projects", []):
            project = _map_project(row)
            if project.project_id:
                projects.append(project)
    return projects


def _map_project(row: dict[str, Any]) -> ProjectContext:
    return ProjectContext(
        projectId=_first(row, "projectId", "ProjectID", "id") or "",
        projectName=_first(row, "projectName", "ProjectName", "projectNames") or "",
        description=_text_value(
            _first(row, "description", "ProjectDescription", "projectDescription")
        ),
        associatedPeople=[
            _map_person(person)
            for person in (_first(row, "AssociatedPeople", "associatedPeople") or [])
            if isinstance(person, dict)
        ],
        associatedManagers=[
            _map_person(person)
            for person in (_first(row, "AssociatedManagers", "associatedManagers") or [])
            if isinstance(person, dict)
        ],
        scopes=[
            _map_scope(scope)
            for scope in (_first(row, "ProjectScopes", "projectScopes", "scopes") or [])
            if isinstance(scope, dict)
        ],
    )


def _map_person(row: dict[str, Any]) -> AssociatedPerson:
    return AssociatedPerson(
        personId=str(_first(row, "AssociatedPersonID", "associatedPersonId", "personId") or ""),
        name=_first(row, "PersonName", "personName", "displayName", "name") or "",
        aliases=_first(row, "PersonAliases", "personAliases", "aliases") or "",
        email=_first(row, "PersonEmail", "personEmail", "email") or "",
        role=_first(row, "Role", "role") or "",
    )


def _map_scope(row: dict[str, Any]) -> ProjectScope:
    return ProjectScope(
        scopeId=_first(row, "ScopeID", "scopeID", "scopeId") or "",
        title=_first(row, "ProjectScopeAreaTitle", "scopeTitle", "title") or "",
        description=_text_value(_first(row, "ProjectScopeArea", "scopeDescription", "description")),
        groupTaskSetId=_first(row, "GroupTaskSetID", "groupTaskSetId"),
    )


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("Text") or value.get("text") or value.get("Description") or ""
    return str(value)


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    return {
        "NO_TASK_FOUND": "no_task_found",
        "NO_PROJECT_MATCH": "no_project_match",
        "TASKS_READY": "tasks_ready",
        "WRITTEN": "written",
        "FAILED": "failed",
        "RETRYABLE": "retryable",
    }.get(str(value), str(value))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Random Synthetic Email Evaluation - {summary['evaluationRunId']}",
        "",
        f"- Total: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['passRate']:.2%}",
        f"- Actual statuses: {summary['actualStatuses']}",
        f"- Model fallback used: {summary['modelFallbackUsed']}",
        "",
        "## By Scenario",
    ]
    for scenario, bucket in summary["byScenario"].items():
        lines.append(
            f"- {scenario}: {bucket['passed']}/{bucket['total']} "
            f"({bucket['passRate']:.2%})"
        )
    lines.extend(["", "## Failure Counts"])
    for failure, count in summary["failureCounts"].items():
        lines.append(f"- {failure}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


if __name__ == "__main__":
    main()
