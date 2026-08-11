"""Provider selection.

One branch per provider. Adding one means writing an adapter that satisfies
LLMClient and adding a case here; nothing else in the service changes.
"""

import logging

import httpx

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def build_llm_client(
    settings: Settings, http_client: httpx.AsyncClient
) -> LLMClient | None:
    """Return a configured client, or None to run planner-free.

    Returning None rather than raising keeps the service fully usable with no
    credentials: the deterministic planner still answers supported phrasings.
    """
    provider = (settings.llm_provider or "").strip().lower()

    if provider in ("", "none", "rulebased"):
        logger.info("LLM disabled; using the deterministic planner only")
        return None

    if provider == "openai":
        key = (settings.openai_api_key or "").strip()
        # Guard the placeholder shipped in .env.example so a half-configured
        # environment degrades to the deterministic planner instead of sending
        # doomed requests on every query.
        if not key or key.startswith("sk-REPLACE"):
            logger.warning(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is unset; "
                "falling back to the deterministic planner"
            )
            return None
        logger.info("Using OpenAI planner (model=%s)", settings.openai_model)
        return OpenAIClient(
            http_client,
            api_key=key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    logger.warning(
        "Unknown LLM_PROVIDER '%s'; using the deterministic planner", provider
    )
    return None
