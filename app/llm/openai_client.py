"""OpenAI adapter using strict structured outputs.

Talks to the REST API over the shared httpx pool rather than the vendor SDK:
the surface used here is a single POST, and going direct keeps the dependency
list small and makes the boundary trivial to stub in tests with MockTransport.
"""

import json
import logging

import httpx
from pydantic import ValidationError

from app.llm.base import LLMClient, LLMError, ModelT, to_strict_schema

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._http = http_client
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

    async def structured(
        self,
        schema: type[ModelT],
        system: str,
        user: str,
        *,
        repair_hint: str | None = None,
    ) -> ModelT:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Deterministic planning: the same question should map to the same
            # plan across calls.
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": to_strict_schema(schema),
                },
            },
        }

        try:
            response = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            # Network-level failure: re-prompting cannot help.
            raise LLMError(f"OpenAI request failed: {exc}", repairable=False) from exc

        if response.status_code != 200:
            # Provider error bodies echo back request context -- including a
            # partially masked API key on 401 -- and LLMError messages surface
            # in client-facing error details. Log the body server-side; let
            # only the status code cross the boundary.
            logger.warning(
                "OpenAI returned HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            raise LLMError(
                f"OpenAI returned HTTP {response.status_code}", repairable=False
            )

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {exc}", repairable=False)

        if content is None:
            # Strict mode refuses rather than emitting non-conforming JSON.
            raise LLMError("OpenAI returned no content", repairable=True)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"OpenAI returned malformed JSON: {exc}", repairable=True)

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            # Schema mismatch is exactly the case a repair prompt can fix.
            raise LLMError(
                f"Plan failed validation: {_summarize(exc)}", repairable=True
            )


def _summarize(exc: ValidationError, limit: int = 3) -> str:
    """Condense a ValidationError into a short, promptable hint."""
    parts = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
