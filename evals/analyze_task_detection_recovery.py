from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from taslow_email_extraction_agent.executors.task_detection import _task_recovery_reason
from taslow_email_extraction_agent.models import EmailExtractionRequest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_case_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["caseId"]): row for row in rows}


def _actual_status(raw_row: dict[str, Any]) -> str:
    response = raw_row.get("response") or {}
    return str(response.get("status") or "")


def analyze(
    requests_path: Path,
    answers_path: Path,
    raw_responses_path: Path,
    selection_path: Path,
    include_details: bool = False,
) -> dict[str, Any]:
    requests = _by_case_id(_load_jsonl(requests_path))
    answers = _by_case_id(_load_jsonl(answers_path))
    raw_responses = _by_case_id(_load_jsonl(raw_responses_path))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    case_ids = [str(case_id) for case_id in selection["caseIds"]]

    role_counts: Counter[str] = Counter()
    attempt_counts: Counter[str] = Counter()
    reason_counts: dict[str, Counter[str]] = {}
    details: list[dict[str, str]] = []
    for case_id in case_ids:
        request_row = requests[case_id]
        answer_row = answers[case_id]
        raw_row = raw_responses[case_id]
        expected_status = str(answer_row.get("expectedStatus") or "")
        actual_status = _actual_status(raw_row)
        if expected_status == "tasks_found" and actual_status != "tasks_found":
            role = "status_false_negative"
        elif expected_status == "no_task_found" and actual_status == "no_task_found":
            role = "matched_no_task_control"
        else:
            role = "other"

        reason = _task_recovery_reason(EmailExtractionRequest.model_validate(request_row))
        role_counts[role] += 1
        if reason:
            attempt_counts[role] += 1
            reason_counts.setdefault(role, Counter())[reason] += 1
            if include_details:
                details.append(
                    {
                        "caseId": case_id,
                        "role": role,
                        "reason": reason,
                        "subject": str(request_row.get("subject") or ""),
                        "bodyExcerpt": " ".join(
                            str(request_row.get("bodyText") or "").split()
                        )[:240],
                    }
                )

    result = {
        "schemaVersion": "1.0.0",
        "selectionId": selection["selectionId"],
        "selectionCaseCount": len(case_ids),
        "roles": {
            role: {
                "caseCount": count,
                "recoveryAttemptCount": attempt_counts[role],
                "recoveryAttemptRate": round(attempt_counts[role] / count, 4),
                "reasons": dict(sorted(reason_counts.get(role, Counter()).items())),
            }
            for role, count in sorted(role_counts.items())
        },
    }
    if include_details:
        result["details"] = details
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the bounded task-detection recovery guard against a sealed selection."
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--raw-responses", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--include-details", action="store_true")
    args = parser.parse_args()

    result = analyze(
        args.requests,
        args.answers,
        args.raw_responses,
        args.selection,
        args.include_details,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
