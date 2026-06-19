from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the extraction service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    azure_ai_project_endpoint: str | None = Field(default=None, alias="AZURE_AI_PROJECT_ENDPOINT")
    azure_ai_model_deployment_name: str | None = Field(
        default=None, alias="AZURE_AI_MODEL_DEPLOYMENT_NAME"
    )
    azure_openai_chat_api_version: str = Field(
        default="2024-08-01-preview", alias="AZURE_OPENAI_CHAT_API_VERSION"
    )
    agent_task_extractor_provider: str = Field(
        default="foundry", alias="AGENT_TASK_EXTRACTOR_PROVIDER"
    )
    agent_task_extractor_fallback_enabled: bool = Field(
        default=True, alias="AGENT_TASK_EXTRACTOR_FALLBACK_ENABLED"
    )
    project_service_base_url: str | None = Field(default=None, alias="PROJECT_SERVICE_BASE_URL")
    task_service_base_url: str | None = Field(default=None, alias="TASK_SERVICE_BASE_URL")
    taslow_service_api_key: str | None = Field(default=None, alias="TASLOW_SERVICE_API_KEY")

    project_search_provider: str = Field(
        default="cosmos-legacy", alias="AGENT_PROJECT_SEARCH_PROVIDER"
    )
    azure_search_endpoint: str | None = Field(default=None, alias="AZURE_SEARCH_ENDPOINT")
    azure_search_index_name: str | None = Field(default=None, alias="AZURE_SEARCH_INDEX_NAME")
    azure_search_api_key: str | None = Field(default=None, alias="AZURE_SEARCH_API_KEY")
    azure_search_api_version: str = Field(default="2024-07-01", alias="AZURE_SEARCH_API_VERSION")
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    azure_openai_embedding_api_version: str = Field(
        default="2024-02-01", alias="AZURE_OPENAI_EMBEDDING_API_VERSION"
    )
    azure_openai_embedding_dimensions: int = Field(
        default=1536, alias="AZURE_OPENAI_EMBEDDING_DIMENSIONS"
    )
    project_search_top_k: int = Field(default=8, alias="PROJECT_SEARCH_TOP_K")
    scope_search_top_k: int = Field(default=5, alias="SCOPE_SEARCH_TOP_K")
    agent_search_dependency_retry_enabled: bool = Field(
        default=True, alias="AGENT_SEARCH_DEPENDENCY_RETRY_ENABLED"
    )
    agent_scope_web_grounding_enabled: bool = Field(
        default=False, alias="AGENT_SCOPE_WEB_GROUNDING_ENABLED"
    )
    agent_scope_web_grounding_provider: str = Field(
        default="none", alias="AGENT_SCOPE_WEB_GROUNDING_PROVIDER"
    )
    agent_scope_web_grounding_max_queries: int = Field(
        default=0, alias="AGENT_SCOPE_WEB_GROUNDING_MAX_QUERIES"
    )

    project_confidence_threshold: float = Field(default=0.80, alias="PROJECT_CONFIDENCE_THRESHOLD")
    scope_confidence_threshold: float = Field(default=0.75, alias="SCOPE_CONFIDENCE_THRESHOLD")
    assignee_confidence_threshold: float = Field(
        default=0.80, alias="ASSIGNEE_CONFIDENCE_THRESHOLD"
    )
    due_date_confidence_threshold: float = Field(
        default=0.70, alias="DUE_DATE_CONFIDENCE_THRESHOLD"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
