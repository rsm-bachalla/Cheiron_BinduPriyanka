"""Planner tests, focused on the refusal contract.

The service must decline ambiguous questions rather than answer a question the
caller did not ask, so the negative cases matter as much as the positive ones.
"""

import pytest

from app.errors import UnsupportedQueryError
from app.planner import plan_query
from app.schemas.api import QueryHints
from app.schemas.plan import AnalysisOp, Dimension, TrialPhase, TrialStatus


class TestRecognisedQueries:
    def test_phase_distribution(self):
        plan = plan_query("How are breast cancer trials distributed across phases?")
        assert plan.operation is AnalysisOp.DISTRIBUTION
        assert plan.dimension is Dimension.PHASE
        assert plan.filters.condition == "breast cancer"

    def test_longest_condition_match_wins(self):
        plan = plan_query(
            "Breakdown of non-small cell lung cancer trials by phase"
        )
        assert plan.filters.condition == "non-small cell lung cancer"

    def test_status_becomes_a_filter(self):
        plan = plan_query("How many recruiting lung cancer trials by country?")
        assert plan.filters.status is TrialStatus.RECRUITING
        assert plan.dimension is Dimension.COUNTRY

    def test_phase_word_is_axis_not_filter_when_grouping_by_phase(self):
        # "across phases" names the grouping; treating it as a filter would
        # collapse the chart to a single bar.
        plan = plan_query("How are melanoma trials distributed across phases?")
        assert plan.filters.phase is None

    def test_phase_is_a_filter_when_grouping_by_something_else(self):
        plan = plan_query("Breakdown of phase 3 melanoma trials by country")
        assert plan.filters.phase is TrialPhase.PHASE3
        assert plan.dimension is Dimension.COUNTRY

    def test_year_range_extracted(self):
        plan = plan_query("Breakdown of melanoma trials by phase from 2018 to 2021")
        assert (plan.filters.start_year, plan.filters.end_year) == (2018, 2021)


class TestRefusals:
    def test_refuses_when_no_grouping_dimension(self):
        with pytest.raises(UnsupportedQueryError) as exc:
            plan_query("Tell me about breast cancer trials")
        assert "group" in exc.value.message.lower()

    def test_refuses_when_no_analysis_cue(self):
        with pytest.raises(UnsupportedQueryError):
            plan_query("phase")

    def test_refuses_unfiltered_query_that_would_scan_registry(self):
        with pytest.raises(UnsupportedQueryError) as exc:
            plan_query("How are trials distributed across phases?")
        assert "entire registry" in exc.value.details["reason"]

    def test_refusal_explains_what_is_supported(self):
        # A refusal that does not tell the caller how to succeed is a dead end.
        with pytest.raises(UnsupportedQueryError) as exc:
            plan_query("What is the meaning of life?")
        assert "OPENAI_API_KEY" in exc.value.details["supported"]

    def test_refusal_carries_structured_details_not_just_a_string(self):
        with pytest.raises(UnsupportedQueryError) as exc:
            plan_query("something entirely unrelated")
        payload = exc.value.to_payload()
        assert payload["error"]["code"] == "unsupported_query"
        assert "reason" in payload["error"]["details"]


class TestHints:
    def test_hint_overrides_inferred_filter(self):
        plan = plan_query(
            "How are breast cancer trials distributed across phases?",
            QueryHints(condition="melanoma"),
        )
        assert plan.filters.condition == "melanoma"

    def test_hints_can_rescue_an_otherwise_unfiltered_query(self):
        plan = plan_query(
            "How are trials distributed across phases?",
            QueryHints(condition="melanoma"),
        )
        assert plan.filters.condition == "melanoma"

    def test_unset_hints_do_not_clobber_inferred_values(self):
        plan = plan_query(
            "How are breast cancer trials distributed across phases?",
            QueryHints(country="France"),
        )
        assert plan.filters.condition == "breast cancer"
        assert plan.filters.country == "France"
