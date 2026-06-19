from __future__ import annotations

from taslow_email_extraction_agent.agent_framework_compat import step
from taslow_email_extraction_agent.models import (
    ExtractedTaskCandidate,
    ProjectContext,
    ProjectScope,
    ThreadContext,
)
from taslow_email_extraction_agent.text_utils import lexical_similarity, token_set


@step(name="ScopeAreaMatchingExecutor")
async def match_scope_area(
    task: ExtractedTaskCandidate,
    project: ProjectContext,
    thread_context: ThreadContext | None,
) -> tuple[ProjectScope | None, float, list[str]]:
    if not project.scopes:
        return None, 0.0, []

    best_scope: ProjectScope | None = None
    best_score = 0.0
    best_evidence: list[str] = []

    for scope in project.scopes:
        lexical = lexical_similarity(task.description, scope.combined_text)
        title_tokens = _significant_tokens(scope.title)
        task_tokens = token_set(task.description)
        alias_hits = title_tokens & task_tokens
        if title_tokens and title_tokens.issubset(task_tokens):
            lexical = max(lexical, 1.0)
        thread = 0.0
        evidence: list[str] = []
        if lexical:
            evidence.append("task_scope_similarity")
        if alias_hits:
            evidence.append("scope_alias_hit")
        semantic = scope.search_score or 0.0
        search_margin = scope.search_margin or 0.0
        if semantic:
            evidence.append("azure_ai_search_scope_similarity")
        if scope.search_rank == 1 and semantic:
            evidence.append("top_scope_search_candidate")
        if thread_context and thread_context.scope_id == scope.scope_id:
            thread = min(1.0, thread_context.confidence)
            evidence.append("thread_scope_history")

        if scope.search_rank == 1 and semantic >= 0.68:
            score = min(
                1.0,
                max(
                    0.80,
                    (semantic * 0.70)
                    + (lexical * 0.20)
                    + (thread * 0.05)
                    + (min(1.0, search_margin * 5) * 0.05),
                ),
            )
        elif semantic:
            score = min(1.0, (semantic * 0.60) + (lexical * 0.30) + (thread * 0.10))
        else:
            score = min(1.0, (lexical * 0.75) + (thread * 0.25))
        if alias_hits and semantic >= 0.55:
            score = max(score, 0.78)
        if score > best_score:
            best_scope = scope
            best_score = score
            best_evidence = evidence

    return best_scope, round(best_score, 3), best_evidence


def _significant_tokens(value: str) -> set[str]:
    stop_words = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "scope",
        "area",
        "task",
        "support",
        "services",
    }
    return {token for token in token_set(value) if len(token) > 2 and token not in stop_words}
