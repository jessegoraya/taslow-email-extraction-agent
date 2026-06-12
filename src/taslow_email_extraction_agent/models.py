from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExtractionStatus(StrEnum):
    NO_TASK_FOUND = "no_task_found"
    NO_PROJECT_MATCH = "no_project_match"
    TASKS_READY = "tasks_ready"
    WRITTEN = "written"
    RETRYABLE = "retryable"
    FAILED = "failed"


class EmailDirection(StrEnum):
    SENT = "sent"
    RECEIVED = "received"


class Participant(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: str = ""
    name: str = ""

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class EmailExtractionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tenant_id: str = Field(alias="tenantId")
    mailbox: str
    direction: EmailDirection = EmailDirection.SENT
    graph_event_id: str = Field(alias="graphEventId")
    internet_message_id: str = Field(alias="internetMessageId")
    message_id: str = Field(alias="messageId")
    subject: str = ""
    body_text: str = Field(default="", alias="bodyText")
    sent_date_time: datetime | None = Field(default=None, alias="sentDateTime")
    from_participant: Participant | None = Field(default=None, alias="from")
    to: list[Participant] = Field(default_factory=list)
    cc: list[Participant] = Field(default_factory=list)
    bcc: list[Participant] = Field(default_factory=list)
    conversation_id: str | None = Field(default=None, alias="conversationId")
    parent_message_id: str | None = Field(default=None, alias="parentMessageId")
    idempotency_key: str = Field(default="", alias="idempotencyKey")
    correlation_id: str = Field(default="", alias="correlationId")

    @field_validator("tenant_id", "mailbox", "graph_event_id", "internet_message_id", "message_id")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field is required")
        return cleaned

    @field_validator("mailbox")
    @classmethod
    def normalize_mailbox(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def subject_or_body_required(self) -> EmailExtractionRequest:
        if not self.subject.strip() and not self.body_text.strip():
            raise ValueError("subject or bodyText is required")
        return self

    @property
    def all_recipients(self) -> list[Participant]:
        return [*self.to, *self.cc, *self.bcc]

    @property
    def visible_recipients(self) -> list[Participant]:
        return [*self.to, *self.cc]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(part for part in [self.subject.strip(), self.body_text.strip()] if part)


class AssociatedPerson(BaseModel):
    person_id: str | None = Field(default=None, alias="personId")
    name: str = ""
    aliases: str = ""
    email: str = ""
    role: str = ""

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def tokens(self) -> set[str]:
        values = [self.name, self.email, self.aliases]
        tokens: set[str] = set()
        for value in values:
            for chunk in value.replace(";", ",").split(","):
                cleaned = chunk.strip().lower()
                if cleaned:
                    tokens.add(cleaned)
                    if "@" in cleaned:
                        tokens.add(cleaned.split("@", 1)[0])
                    for word in cleaned.split():
                        tokens.add(word)
        return tokens


class ProjectScope(BaseModel):
    scope_id: str = Field(alias="scopeId")
    title: str = ""
    description: str = ""
    embeddings: list[float] = Field(default_factory=list)
    group_task_set_id: str | None = Field(default=None, alias="groupTaskSetId")
    search_score: float | None = Field(default=None, alias="searchScore")

    @property
    def combined_text(self) -> str:
        return " ".join(part for part in [self.title, self.description] if part).strip()


class ProjectContext(BaseModel):
    project_id: str = Field(alias="projectId")
    project_name: str = Field(default="", alias="projectName")
    description: str = ""
    desc_vector: list[float] = Field(default_factory=list, alias="descVector")
    search_score: float | None = Field(default=None, alias="searchScore")
    associated_people: list[AssociatedPerson] = Field(
        default_factory=list, alias="associatedPeople"
    )
    associated_managers: list[AssociatedPerson] = Field(
        default_factory=list, alias="associatedManagers"
    )
    scopes: list[ProjectScope] = Field(default_factory=list)

    @property
    def people(self) -> list[AssociatedPerson]:
        return [*self.associated_people, *self.associated_managers]

    @property
    def combined_text(self) -> str:
        return " ".join(part for part in [self.project_name, self.description] if part).strip()


class ThreadContext(BaseModel):
    project_id: str | None = Field(default=None, alias="projectId")
    scope_id: str | None = Field(default=None, alias="scopeId")
    confidence: float = 0.0


class ExtractedTaskCandidate(BaseModel):
    source_task_id: str = Field(alias="sourceTaskId")
    title: str
    description: str
    mentioned_people: list[str] = Field(default_factory=list, alias="mentionedPeople")
    due_text: str | None = Field(default=None, alias="dueText")
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class ProjectMatchResult(BaseModel):
    project_id: str | None = Field(default=None, alias="projectId")
    project_name: str | None = Field(default=None, alias="projectName")
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class ExtractedTaskAssignment(BaseModel):
    source_task_id: str = Field(alias="sourceTaskId")
    title: str
    description: str
    project_id: str = Field(alias="projectId")
    scope_id: str | None = Field(default=None, alias="scopeId")
    scope_confidence: float | None = Field(default=None, alias="scopeConfidence")
    assignee_email: str = Field(alias="assigneeEmail")
    assignee_name: str = Field(default="", alias="assigneeName")
    assignee_confidence: float = Field(default=0.0, alias="assigneeConfidence")
    due_date: datetime | None = Field(default=None, alias="dueDate")
    due_date_confidence: float | None = Field(default=None, alias="dueDateConfidence")
    overall_confidence: float = Field(default=0.0, alias="overallConfidence")
    evidence: list[str] = Field(default_factory=list)
    needs_review: bool = Field(default=False, alias="needsReview")


class ExtractionDiagnostics(BaseModel):
    model: str | None = None
    project_threshold: float = Field(alias="projectThreshold")
    scope_threshold: float = Field(alias="scopeThreshold")
    assignee_threshold: float = Field(alias="assigneeThreshold")
    due_date_threshold: float = Field(alias="dueDateThreshold")
    stopped_after: str | None = Field(default=None, alias="stoppedAfter")
    warnings: list[str] = Field(default_factory=list)
    retry_schedule: list[str] = Field(default_factory=list, alias="retrySchedule")
    manual_execution_required: bool = Field(default=False, alias="manualExecutionRequired")


class EmailExtractionResponse(BaseModel):
    agent_run_id: str = Field(alias="agentRunId")
    status: ExtractionStatus
    tenant_id: str = Field(alias="tenantId")
    graph_event_id: str = Field(alias="graphEventId")
    internet_message_id: str = Field(alias="internetMessageId")
    message_id: str = Field(alias="messageId")
    task_candidate_count: int = Field(default=0, alias="taskCandidateCount")
    project_match: ProjectMatchResult | None = Field(default=None, alias="projectMatch")
    tasks: list[ExtractedTaskAssignment] = Field(default_factory=list)
    diagnostics: ExtractionDiagnostics

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json", exclude_none=True)
