"""Planner orchestration.

    query -> OpenAI structured output -> Pydantic validation
          -> semantic validation -> merge hints -> QueryPlan

with a bounded failure ladder: at most one repair attempt, then the
deterministic pattern planner, then a structured refusal. The service never
guesses an interpretation it cannot justify.
"""

import logging
from dataclasses import dataclass

from app.errors import UnsupportedQueryError
from app.llm.base import LLMClient, LLMError
from app.llm.prompts import PLANNER_SYSTEM_PROMPT, build_user_prompt
from app.plan_validation import PlanValidationError, validate_plan
from app.planner import apply_hints, plan_query as plan_query_rulebased
from app.schemas.api import QueryHints
from app.schemas.plan import QueryPlan

logger = logging.getLogger(__name__)


@dataclass
class PlanningOutcome:
    """A plan plus how it was reached, for meta reporting and debugging."""

    plan: QueryPlan
    planner: str  # "openai" | "rulebased"
    repaired: bool = False
    fallback_reason: str | None = None


async def _plan_with_llm(
    llm: LLMClient, query: str
) -> tuple[QueryPlan, bool]:
    """One LLM attempt, plus at most one repair. Returns (plan, repaired)."""
    repair_hint: str | None = None

    for attempt in range(2):
        is_repair = attempt == 1
        plan = await llm.structured(
            QueryPlan,
            PLANNER_SYSTEM_PROMPT,
            build_user_prompt(query, repair_hint),
            repair_hint=repair_hint,
        )
        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            if is_repair or not exc.repairable:
                # Either we already spent the repair, or re-prompting cannot
                # fix it (e.g. the question named no subject at all).
                raise
            logger.info("Plan rejected (%s); attempting one repair", exc.reason)
            repair_hint = exc.reason
            continue

        return plan, is_repair

    raise AssertionError("unreachable")  # pragma: no cover


async def plan_query(
    query: str,
    hints: QueryHints | None = None,
    llm: LLMClient | None = None,
) -> PlanningOutcome:
    """Produce a validated, hint-merged plan, or raise UnsupportedQueryError."""
    llm_failure: str | None = None

    if llm is not None:
        try:
            plan, repaired = await _plan_with_llm(llm, query)
        except LLMError as exc:
            # Provider unavailable, or output we could not repair.
            llm_failure = exc.message
            logger.warning("LLM planner failed (%s); trying fallback", exc.message)
        except PlanValidationError as exc:
            llm_failure = exc.reason
            logger.warning(
                "LLM plan failed semantic validation (%s); trying fallback", exc.reason
            )
        else:
            # Hints are applied after validation so an explicit caller filter
            # always wins over whatever the model inferred.
            plan = plan.model_copy(
                update={"filters": apply_hints(plan.filters, hints)}
            )
            logger.info(
                "Planned via openai (repaired=%s): operation=%s dimension=%s",
                repaired,
                plan.operation.value,
                plan.dimension.value,
            )
            return PlanningOutcome(plan=plan, planner="openai", repaired=repaired)

    # Fallback: only for queries the deterministic planner matches confidently.
    # It raises UnsupportedQueryError itself when it does not, which is the
    # refusal the caller receives.
    try:
        plan = plan_query_rulebased(query, hints)
    except UnsupportedQueryError as exc:
        if llm_failure:
            # Make it clear the refusal followed an LLM failure, so a caller can
            # distinguish "we cannot parse this" from "the provider was down".
            exc.details["llm_error"] = llm_failure
        raise

    logger.info(
        "Planned via rulebased fallback: operation=%s dimension=%s",
        plan.operation.value,
        plan.dimension.value,
    )
    return PlanningOutcome(
        plan=plan, planner="rulebased", fallback_reason=llm_failure
    )
