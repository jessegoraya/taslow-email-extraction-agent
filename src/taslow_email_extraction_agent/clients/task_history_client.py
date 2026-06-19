from __future__ import annotations

from typing import Protocol

from taslow_email_extraction_agent.models import EmailExtractionRequest, ThreadContext


class TaskHistoryClient(Protocol):
    async def get_thread_context(self, request: EmailExtractionRequest) -> ThreadContext | None:
        """Return prior project/scope context for a related email thread."""


class EmptyTaskHistoryClient:
    async def get_thread_context(self, request: EmailExtractionRequest) -> ThreadContext | None:
        return None


class InMemoryTaskHistoryClient:
    def __init__(self, by_conversation_id: dict[str, ThreadContext] | None = None) -> None:
        self._by_conversation_id = by_conversation_id or {}

    async def get_thread_context(self, request: EmailExtractionRequest) -> ThreadContext | None:
        if request.conversation_id and request.conversation_id in self._by_conversation_id:
            return self._by_conversation_id[request.conversation_id]
        if request.parent_message_id and request.parent_message_id in self._by_conversation_id:
            return self._by_conversation_id[request.parent_message_id]
        if request.internet_message_id and request.internet_message_id in self._by_conversation_id:
            return self._by_conversation_id[request.internet_message_id]
        if request.message_id and request.message_id in self._by_conversation_id:
            return self._by_conversation_id[request.message_id]
        return None

    def record_thread_context(
        self,
        request: EmailExtractionRequest,
        context: ThreadContext,
    ) -> None:
        for key in [
            request.conversation_id,
            request.parent_message_id,
            request.internet_message_id,
            request.message_id,
        ]:
            if key:
                self._by_conversation_id[key] = context
