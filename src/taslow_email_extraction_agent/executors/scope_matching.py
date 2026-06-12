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
        title_tokens = token_set(scope.title)
        task_tokens = token_set(task.description)
        if title_tokens and title_tokens.issubset(task_tokens):
            lexical = max(lexical, 1.0)
        thread = 0.0
        evidence: list[str] = []
        if lexical:
            evidence.append("task_scope_similarity")
        semantic = scope.search_score or 0.0
        if semantic:
            evidence.append("azure_ai_search_scope_similarity")
        if thread_context and thread_context.scope_id == scope.scope_id:
            thread = min(1.0, thread_context.confidence)
            evidence.append("thread_scope_history")

        if semantic:
            score = min(1.0, (lexical * 0.55) + (semantic * 0.25) + (thread * 0.20))
        else:
            score = min(1.0, (lexical * 0.75) + (thread * 0.25))
        if score > best_score:
            best_scope = scope
            best_score = score
            best_evidence = evidence

    return best_scope, round(best_score, 3), best_evidence
