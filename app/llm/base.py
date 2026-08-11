"""Provider-agnostic structured-generation boundary.

The entire LLM surface of this service is one method: given a Pydantic model and
a prompt, return an instance of that model. Adding a provider means writing one
adapter, not touching the planner.

Only OpenAI is implemented for this submission.
"""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(Exception):
    """Provider call failed. Carries whether a retry could plausibly help."""

    def __init__(self, message: str, *, repairable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        # True for schema/validation mismatches, where re-prompting with the
        # error may succeed. False for auth, network, and rate-limit failures,
        # where an immediate retry only adds latency.
        self.repairable = repairable


class LLMClient(Protocol):
    """Structured generation against a JSON schema."""

    name: str

    async def structured(
        self,
        schema: type[ModelT],
        system: str,
        user: str,
        *,
        repair_hint: str | None = None,
    ) -> ModelT:
        """Return a validated `schema` instance, or raise LLMError."""
        ...


# JSON Schema keywords OpenAI's strict structured-output mode rejects. Pydantic
# emits several of them (from Field constraints), so they are stripped before
# the schema is sent. The constraints they express are enforced by Pydantic on
# the way back in, and by semantic validation after that.
_UNSUPPORTED_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "default",
    "minItems",
    "maxItems",
}


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an OpenAI strict-mode JSON schema.

    Strict mode requires every object to list all of its properties as required
    and to set `additionalProperties: false`. Optional fields stay expressible
    because Pydantic renders them as `anyOf: [..., {"type": "null"}]`, which
    strict mode does allow.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        cleaned = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}

        if cleaned.get("type") == "object" and "properties" in cleaned:
            cleaned["additionalProperties"] = False
            # Strict mode has no notion of an omitted key; optionality is
            # carried by the null branch of the type union instead.
            cleaned["required"] = list(cleaned["properties"].keys())

        return cleaned

    return walk(model.model_json_schema())
