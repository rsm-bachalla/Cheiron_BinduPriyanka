"""LLM planner orchestration tests.

The OpenAI boundary is stubbed throughout -- no network, no real LLM calls. What
is under test is the flow around the model: validation, repair, hint merging,
and the fallback ladder.
"""

import pytest
from pydantic import ValidationError

from app.errors import UnsupportedQueryError
from app.llm.base import LLMError
from app.plan_validation import PlanValidationError, validate_plan
from app.planning import plan_query
from app.schemas.api import QueryHints
from app.schemas.plan import AnalysisOp, Dimension, QueryPlan, TrialStatus


class StubLLM:
    """Returns queued plans or raises queued errors, recording every call."""

    name = "stub"

    def __init__(self, *results):
        self._results = list(results)
        self.calls: list[dict] = []

    async def structured(self, schema, system, user, *, repair_hint=None):
        self.calls.append({"user": user, "repair_hint": repair_hint})
        if not self._results:
            raise AssertionError("StubLLM called more times than expected")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result)


def _plan(**overrides) -> dict:
    base = {
        "intent": "Break down breast cancer trials by phase.",
        "operation": "distribution",
        "dimension": "phase",
        "filters": {"condition": "breast cancer"},
        "comparison_groups": [],
        "title": "Breast Cancer Trials by Phase",
    }
    base.update(overrides)
    return base


class TestRepresentativeQueries:
    """The five assignment example queries, end to end through the planner."""

    async def test_distribution(self):
        llm = StubLLM(_plan())
        outcome = await plan_query(
            "How are breast cancer trials distributed across phases?", llm=llm
        )
        assert outcome.planner == "openai"
        assert outcome.plan.operation is AnalysisOp.DISTRIBUTION
        assert outcome.plan.dimension is Dimension.PHASE
        assert outcome.plan.filters.condition == "breast cancer"

    async def test_time_trend(self):
        llm = StubLLM(
            _plan(
                operation="time_trend",
                dimension="year",
                filters={"drug": "pembrolizumab"},
            )
        )
        outcome = await plan_query(
            "How has the number of pembrolizumab trials changed over time?", llm=llm
        )
        assert outcome.plan.operation is AnalysisOp.TIME_TREND
        assert outcome.plan.dimension is Dimension.YEAR
        assert outcome.plan.filters.drug == "pembrolizumab"

    async def test_geo_ranking(self):
        llm = StubLLM(
            _plan(
                operation="geo",
                dimension="country",
                filters={"condition": "lung cancer", "status": "RECRUITING"},
            )
        )
        outcome = await plan_query(
            "Which countries have the most recruiting lung cancer trials?", llm=llm
        )
        assert outcome.plan.operation is AnalysisOp.GEO
        assert outcome.plan.filters.status is TrialStatus.RECRUITING

    async def test_comparison(self):
        llm = StubLLM(
            _plan(
                operation="comparison",
                dimension="phase",
                filters={},
                comparison_groups=["pembrolizumab", "nivolumab"],
            )
        )
        outcome = await plan_query(
            "Compare trial phases for pembrolizumab vs nivolumab", llm=llm
        )
        assert outcome.plan.operation is AnalysisOp.COMPARISON
        assert outcome.plan.comparison_groups == ["pembrolizumab", "nivolumab"]

    async def test_network(self):
        llm = StubLLM(
            _plan(
                operation="network",
                dimension="sponsor",
                filters={"condition": "melanoma"},
            )
        )
        outcome = await plan_query(
            "Show a network of sponsors and drugs for melanoma trials", llm=llm
        )
        assert outcome.plan.operation is AnalysisOp.NETWORK
        assert outcome.plan.dimension is Dimension.SPONSOR


class TestHintPrecedence:
    async def test_hint_overrides_llm_extracted_filter(self):
        # The documented contract: an explicit hint always wins.
        llm = StubLLM(
            _plan(filters={"condition": "lung cancer", "status": "RECRUITING"})
        )
        outcome = await plan_query(
            "Show recruiting lung cancer trials by phase",
            hints=QueryHints(status=TrialStatus.COMPLETED),
            llm=llm,
        )
        assert outcome.plan.filters.status is TrialStatus.COMPLETED
        assert outcome.plan.filters.condition == "lung cancer"

    async def test_unset_hints_leave_llm_filters_intact(self):
        llm = StubLLM(_plan(filters={"condition": "breast cancer"}))
        outcome = await plan_query(
            "breast cancer trials by phase", hints=QueryHints(), llm=llm
        )
        assert outcome.plan.filters.condition == "breast cancer"


class TestFilterValueSanitisation:
    """Regression guard for prompt artifacts reaching the upstream query.

    Observed live: the model returned `pembrolizumab},` as the drug. The
    upstream API tokenised the junk away and returned the correct studies, so
    the corrupted filter never showed up in the data -- only in the echoed
    meta. The results being right is precisely why this needs a test.
    """

    async def test_strips_json_artifacts_from_extracted_values(self):
        llm = StubLLM(
            _plan(
                operation="time_trend",
                dimension="year",
                filters={"drug": "pembrolizumab},"},
            )
        )
        outcome = await plan_query("pembrolizumab trials over time", llm=llm)
        assert outcome.plan.filters.drug == "pembrolizumab"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('"breast cancer"', "breast cancer"),
            ("{melanoma}", "melanoma"),
            ("  lung   cancer  ", "lung cancer"),
            ("melanoma,", "melanoma"),
            ("[Merck]", "Merck"),
        ],
    )
    def test_cleans_representative_artifacts(self, raw, expected):
        from app.schemas.plan import TrialFilters

        assert TrialFilters(condition=raw).condition == expected

    def test_value_that_is_only_punctuation_becomes_none(self):
        # An empty phrase match would silently match everything.
        from app.schemas.plan import TrialFilters

        assert TrialFilters(condition='{",}').condition is None

    def test_internal_punctuation_is_preserved(self):
        # Only the edges are stripped; real names keep their characters.
        from app.schemas.plan import TrialFilters

        assert (
            TrialFilters(condition="non-small cell lung cancer").condition
            == "non-small cell lung cancer"
        )
        assert TrialFilters(sponsor="Bristol-Myers Squibb").sponsor == (
            "Bristol-Myers Squibb"
        )


class TestSemanticValidation:
    """Checks that schema validation alone cannot catch."""

    def test_time_trend_requires_a_time_dimension(self):
        plan = QueryPlan.model_validate(
            _plan(operation="time_trend", dimension="sponsor")
        )
        with pytest.raises(PlanValidationError, match="cannot group by"):
            validate_plan(plan)

    def test_geo_requires_country(self):
        plan = QueryPlan.model_validate(_plan(operation="geo", dimension="phase"))
        with pytest.raises(PlanValidationError, match="cannot group by"):
            validate_plan(plan)

    def test_comparison_groups_rejected_on_other_operations(self):
        plan = QueryPlan.model_validate(_plan(comparison_groups=["a", "b"]))
        with pytest.raises(PlanValidationError, match="only valid for"):
            validate_plan(plan)

    def test_invalid_year_range_rejected_at_schema_layer(self):
        # Caught by TrialFilters' own validator, i.e. before a plan can exist.
        # An LLM emitting this gets a repairable error and one corrected retry.
        with pytest.raises(ValidationError, match="start_year must not be after"):
            QueryPlan.model_validate(
                _plan(filters={"condition": "x", "start_year": 2022, "end_year": 2019})
            )

    def test_invalid_year_range_also_guarded_semantically(self):
        # Defence in depth: a plan assembled without validation (or a future
        # relaxation of the field validator) must still be rejected here.
        plan = QueryPlan.model_validate(_plan(filters={"condition": "x"}))
        plan.filters.start_year = 2022
        plan.filters.end_year = 2019
        with pytest.raises(PlanValidationError, match="after end_year"):
            validate_plan(plan)

    def test_unfiltered_plan_rejected_and_not_repairable(self):
        # Re-prompting cannot invent a subject the question never named.
        plan = QueryPlan.model_validate(_plan(filters={}))
        with pytest.raises(PlanValidationError) as exc:
            validate_plan(plan)
        assert exc.value.repairable is False

    def test_comparison_exempt_from_the_filter_requirement(self):
        # Its groups become the filters at fetch time.
        plan = QueryPlan.model_validate(
            _plan(operation="comparison", comparison_groups=["a", "b"], filters={})
        )
        validate_plan(plan)  # must not raise


class TestRepair:
    async def test_repairs_once_then_succeeds(self):
        llm = StubLLM(
            _plan(operation="time_trend", dimension="sponsor"),  # incoherent
            _plan(operation="time_trend", dimension="year"),  # corrected
        )
        outcome = await plan_query("trend of melanoma trials", llm=llm)
        assert outcome.planner == "openai"
        assert outcome.repaired is True
        assert len(llm.calls) == 2
        # The repair prompt must name the actual failure.
        assert "cannot group by" in llm.calls[1]["repair_hint"]

    async def test_repairs_at_most_once(self):
        llm = StubLLM(
            _plan(operation="geo", dimension="phase"),
            _plan(operation="geo", dimension="phase"),  # still wrong
        )
        with pytest.raises(UnsupportedQueryError):
            await plan_query("something vague about geography", llm=llm)
        assert len(llm.calls) == 2  # never a third

    async def test_no_repair_when_not_repairable(self):
        # An unfiltered plan cannot be fixed by re-prompting, so do not spend
        # a second call on it.
        llm = StubLLM(_plan(filters={}))
        with pytest.raises(UnsupportedQueryError):
            await plan_query("how are trials distributed across phases", llm=llm)
        assert len(llm.calls) == 1

    async def test_malformed_output_is_repaired_once(self):
        llm = StubLLM(LLMError("malformed JSON", repairable=True), _plan())
        outcome = await plan_query(
            "How are breast cancer trials distributed across phases?", llm=llm
        )
        # The LLMError propagates out of the attempt, so the ladder moves to
        # the deterministic planner rather than re-prompting blindly.
        assert outcome.planner == "rulebased"


class TestFallbackLadder:
    async def test_llm_unavailable_falls_back_for_supported_pattern(self):
        llm = StubLLM(LLMError("connection refused", repairable=False))
        outcome = await plan_query(
            "How are breast cancer trials distributed across phases?", llm=llm
        )
        assert outcome.planner == "rulebased"
        assert outcome.fallback_reason == "connection refused"
        assert outcome.plan.filters.condition == "breast cancer"

    async def test_llm_unavailable_and_ambiguous_query_refuses(self):
        llm = StubLLM(LLMError("service unavailable", repairable=False))
        with pytest.raises(UnsupportedQueryError) as exc:
            await plan_query("tell me something interesting", llm=llm)
        # The refusal must distinguish "unparseable" from "provider was down".
        assert exc.value.details["llm_error"] == "service unavailable"
        assert exc.value.details["reason"]

    async def test_no_llm_configured_uses_deterministic_planner(self):
        outcome = await plan_query(
            "How are breast cancer trials distributed across phases?", llm=None
        )
        assert outcome.planner == "rulebased"
        assert outcome.fallback_reason is None

    async def test_hints_survive_the_fallback_path(self):
        llm = StubLLM(LLMError("down", repairable=False))
        outcome = await plan_query(
            "How are breast cancer trials distributed across phases?",
            hints=QueryHints(country="France"),
            llm=llm,
        )
        assert outcome.plan.filters.country == "France"
