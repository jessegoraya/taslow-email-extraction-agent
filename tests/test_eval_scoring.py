from __future__ import annotations

import json

from evals.run_synthetic_email_eval import _find_answer, _load_answer_key, _score_one


def test_s5_informational_accepts_no_task_found_when_answer_key_says_no_project_match():
    score = _score_one(
        {
            "response": {
                "status": "no_task_found",
                "tasks": [],
                "projectMatch": None,
            }
        },
        {
            "scenarioId": "S5_NON_TASLOW_INFORMATIONAL_NO_TASK",
            "expectedStatus": "no_project_match",
            "shouldCreateTaslowTask": False,
            "expectedTaskCount": 0,
            "expectedTasks": [],
        },
    )

    assert score["passed"] is True
    assert "status" not in score["failures"]


def test_answer_lookup_prefers_exact_message_id_before_source_line_fallback(tmp_path):
    answer_key = tmp_path / "answer_key.jsonl"
    answer_key.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "emailId": "synthetic-message-000047",
                        "expectedProjectId": "wrong-source-line-project",
                    }
                ),
                json.dumps(
                    {
                        "emailId": "synthetic-message-000053",
                        "expectedProjectId": "correct-message-project",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    answer_by_key = _load_answer_key(answer_key)
    answer = _find_answer(
        answer_by_key,
        {
            "messageId": "synthetic-message-000053",
            "graphEventId": "synthetic-graph-event-000053",
            "sourceLine": 47,
        },
    )

    assert answer is not None
    assert answer["expectedProjectId"] == "correct-message-project"


def test_same_project_name_different_id_is_review_flag_not_project_failure():
    score = _score_one(
        {
            "response": {
                "status": "tasks_ready",
                "projectMatch": {
                    "projectId": "actual-project-id",
                    "projectName": "Financial Systems Modernization (FSM)",
                },
                "tasks": [
                    {
                        "scopeId": "scope-1",
                        "assigneeEmail": "jesse@taslow.com",
                    }
                ],
            }
        },
        {
            "expectedStatus": "tasks_ready",
            "expectedProjectId": "expected-project-id",
            "expectedProjectName": "Financial Systems Modernization (FSM)",
            "expectedScopeId": "scope-1",
            "expectedTaskCount": 1,
            "expectedTasks": [
                {
                    "expectedAssigneeEmail": "jesse@taslow.com",
                }
            ],
        },
    )

    assert "project" not in score["failures"]
    assert "same_project_name_different_project_id" in score["reviewFlags"]
    assert score["passed"] is True


def test_unknown_expected_assignee_is_review_flag_not_assignee_failure():
    score = _score_one(
        {
            "response": {
                "status": "tasks_ready",
                "projectMatch": {"projectId": "project-1", "projectName": "IPAWS"},
                "tasks": [
                    {
                        "scopeId": "scope-1",
                        "assigneeEmail": "jesse@taslow.com",
                    }
                ],
            }
        },
        {
            "scenarioId": "S2_TASLOW_IMPLICIT_TASK",
            "subScenarioId": "UNKNOWN_ASSIGNEE",
            "expectedStatus": "tasks_ready",
            "expectedProjectId": "project-1",
            "expectedScopeId": "scope-1",
            "expectedTaskCount": 1,
            "expectedTasks": [
                {
                    "expectedAssigneeEmail": "marcus.devlin@taslow.com",
                }
            ],
        },
        {"project-1": {"jesse@taslow.com"}},
    )

    assert score["passed"] is True
    assert "assignee" not in score["failures"]
    assert "unknown_assignee_expected_assignment_policy_review" in score["reviewFlags"]
    assert "expected_assignee_not_project_associated" in score["reviewFlags"]


def test_external_expected_assignee_is_review_flag_not_assignee_failure():
    score = _score_one(
        {
            "response": {
                "status": "tasks_ready",
                "projectMatch": {"projectId": "project-1", "projectName": "FSM"},
                "tasks": [
                    {
                        "scopeId": "scope-1",
                        "assigneeEmail": "jesse@taslow.com",
                    }
                ],
            }
        },
        {
            "scenarioId": "S2_TASLOW_IMPLICIT_TASK",
            "expectedStatus": "tasks_ready",
            "expectedProjectId": "project-1",
            "expectedScopeId": "scope-1",
            "expectedTaskCount": 1,
            "expectedTasks": [
                {
                    "expectedAssigneeEmail": "ramona.thomas@hq.dhs.gov",
                }
            ],
        },
        {"project-1": {"jesse@taslow.com", "ramona.thomas@hq.dhs.gov"}},
    )

    assert score["passed"] is True
    assert "assignee" not in score["failures"]
    assert "external_expected_assignee_policy_review" in score["reviewFlags"]
