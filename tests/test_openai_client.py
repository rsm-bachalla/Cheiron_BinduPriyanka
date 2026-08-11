"""OpenAI adapter tests, stubbed at the HTTP boundary.

Covers the request contract we send and how each failure mode is classified,
since that classification drives whether the planner spends a repair attempt.
"""

import json

import httpx
import pytest

from app.llm.base import LLMError, to_strict_schema
from app.llm.openai_client import OpenAIClient
from app.schemas.plan import QueryPlan

VALID_PLAN = {
    "intent": "Break down breast cancer trials by phase.",
    "operation": "distribution",
    "dimension": "phase",
    "filters": {
        "drug": None,
        "condition": "breast cancer",
        "sponsor": None,
        "country": None,
        "phase": None,
        "status": None,
        "start_year": None,
        "end_year": None,
    },
    "comparison_groups": [],
    "title": "Breast Cancer Trials by Phase",
}


def _client(handler, **kwargs) -> OpenAIClient:
    return OpenAIClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key="sk-test",
        model="gpt-4o-mini",
        **kwargs,
    )


def _completion(content) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestStrictSchema:
    def test_all_objects_are_closed_and_fully_required(self):
        # OpenAI strict mode rejects schemas that omit either.
        schema = to_strict_schema(QueryPlan)

        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node["additionalProperties"] is False
                    assert set(node["required"]) == set(node["properties"])
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(schema)

    def test_unsupported_keywords_stripped(self):
        # Pydantic emits minimum/maximum from Field(ge=..., le=...), which
        # strict mode rejects. The constraints still apply on the way back in.
        raw = json.dumps(to_strict_schema(QueryPlan))
        assert "minimum" not in raw
        assert "maximum" not in raw

    def test_enums_survive_conversion(self):
        # The enum constraint is the whole point; losing it would let the model
        # return arbitrary operations.
        raw = json.dumps(to_strict_schema(QueryPlan))
        assert "time_trend" in raw
        assert "RECRUITING" in raw


class TestRequestContract:
    async def test_sends_strict_json_schema_and_zero_temperature(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_completion(json.dumps(VALID_PLAN)))

        await _client(handler).structured(QueryPlan, "sys", "usr")
        assert seen["response_format"]["type"] == "json_schema"
        assert seen["response_format"]["json_schema"]["strict"] is True
        # Planning should be reproducible across identical questions.
        assert seen["temperature"] == 0

    async def test_sends_bearer_auth(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json=_completion(json.dumps(VALID_PLAN)))

        await _client(handler).structured(QueryPlan, "sys", "usr")
        assert seen["auth"] == "Bearer sk-test"

    async def test_honours_custom_base_url(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=_completion(json.dumps(VALID_PLAN)))

        await _client(handler, base_url="https://gateway.internal/v1").structured(
            QueryPlan, "sys", "usr"
        )
        assert seen["url"] == "https://gateway.internal/v1/chat/completions"

    async def test_returns_validated_plan(self):
        def handler(request):
            return httpx.Response(200, json=_completion(json.dumps(VALID_PLAN)))

        plan = await _client(handler).structured(QueryPlan, "sys", "usr")
        assert isinstance(plan, QueryPlan)
        assert plan.filters.condition == "breast cancer"


class TestFailureClassification:
    """Whether a failure is `repairable` decides if we spend a second call."""

    async def test_malformed_json_is_repairable(self):
        def handler(request):
            return httpx.Response(200, json=_completion("not json at all"))

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.repairable is True

    async def test_schema_violation_is_repairable(self):
        bad = dict(VALID_PLAN, operation="telepathy")

        def handler(request):
            return httpx.Response(200, json=_completion(json.dumps(bad)))

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.repairable is True
        assert "operation" in exc.value.message

    async def test_refusal_content_is_repairable(self):
        def handler(request):
            return httpx.Response(200, json=_completion(None))

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.repairable is True

    async def test_auth_failure_is_not_repairable(self):
        # Re-prompting a 401 only burns latency.
        def handler(request):
            return httpx.Response(401, text="Incorrect API key provided")

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.repairable is False

    async def test_network_failure_is_not_repairable(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.repairable is False

    async def test_provider_error_body_never_crosses_the_boundary(self):
        # LLMError messages surface in client-facing error details, and OpenAI
        # echoes a partially masked key back on 401. Only the status escapes.
        leaky = (
            '{"error":{"message":"Incorrect API key provided: '
            'sk-inval**************ting"}}'
        )

        def handler(request):
            return httpx.Response(401, text=leaky)

        with pytest.raises(LLMError) as exc:
            await _client(handler).structured(QueryPlan, "sys", "usr")
        assert exc.value.message == "OpenAI returned HTTP 401"
        assert "sk-" not in exc.value.message


class TestFactory:
    def test_placeholder_key_degrades_to_deterministic_planner(self):
        # A half-configured .env must not send doomed requests on every query.
        from app.config import Settings
        from app.llm.factory import build_llm_client

        settings = Settings(llm_provider="openai", openai_api_key="sk-REPLACE_ME")
        assert build_llm_client(settings, httpx.AsyncClient()) is None

    def test_missing_key_degrades_to_deterministic_planner(self):
        from app.config import Settings
        from app.llm.factory import build_llm_client

        settings = Settings(llm_provider="openai", openai_api_key=None)
        assert build_llm_client(settings, httpx.AsyncClient()) is None

    def test_rulebased_provider_returns_none(self):
        from app.config import Settings
        from app.llm.factory import build_llm_client

        settings = Settings(llm_provider="rulebased")
        assert build_llm_client(settings, httpx.AsyncClient()) is None

    def test_valid_key_builds_client(self):
        from app.config import Settings
        from app.llm.factory import build_llm_client

        settings = Settings(llm_provider="openai", openai_api_key="sk-real-key")
        client = build_llm_client(settings, httpx.AsyncClient())
        assert client is not None
        assert client.name == "openai"
