"""Application configuration, loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ClinicalTrials.gov
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    # The API silently caps pageSize at 1000, so never request more than that.
    ctgov_page_size: int = 200
    # Upper bound on studies pulled for a single question. Bounds latency; any
    # truncation is always disclosed in the response meta.
    ctgov_max_records: int = 1000
    ctgov_timeout_seconds: float = 30.0
    ctgov_max_retries: int = 3

    # Citations
    citations_per_point: int = 3

    # LLM. Unused in the deterministic slice; wired in the planner step.
    #
    # "rulebased" uses the deterministic pattern planner only (no credentials
    # needed). "openai" uses OpenAI structured outputs, falling back to the
    # rule-based planner only for patterns it can map with confidence.
    #
    # The planner talks to an LLMClient Protocol, so adding another provider
    # later is a new adapter and one branch in the factory. Only OpenAI is
    # implemented for this submission.
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None  # set for OpenAI-compatible gateways


@lru_cache
def get_settings() -> Settings:
    return Settings()
