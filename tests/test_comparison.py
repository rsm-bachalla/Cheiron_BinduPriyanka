"""Comparison analysis: per-group fan-out, grouped aggregation, per-group meta.

Two layers are exercised separately -- the aggregator over already-tagged
studies, and the HTTP path with the upstream API stubbed per group. No network
calls anywhere.
"""

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as app_module
from app.aggregate import comparison
from app.plan_validation import PlanValidationError, validate_plan
from app.planner import apply_hints
from app.planning import PlanningOutcome
from app.schemas.plan import Dimension, QueryPlan
from app.schemas.study import Study


def _study(nct: str, group: str, **kwargs) -> Study:
    return Study(nct_id=nct, brief_title=f"Trial {nct}", group=group, **kwargs)


GROUPS = ["pembrolizumab", "nivolumab"]


class TestComparisonAggregation:
    def test_two_drugs_across_phase(self):
        studies = [
            _study("NCT1", "pembrolizumab", phases=["PHASE3"]),
            _study("NCT2", "pembrolizumab", phases=["PHASE3"]),
            _study("NCT3", "pembrolizumab", phases=["PHASE1"]),
            _study("NCT4", "nivolumab", phases=["PHASE3"]),
        ]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        counts = {(b.label, b.group): b.value for b in result.buckets}
        assert counts[("Phase 3", "pembrolizumab")] == 2
        assert counts[("Phase 3", "nivolumab")] == 1
        assert counts[("Phase 1", "pembrolizumab")] == 1

    def test_absent_combinations_are_explicit_zeros(self):
        # A grouped bar chart needs the gap to be a zero-height bar, not a
        # missing row the frontend has to infer.
        studies = [_study("NCT1", "pembrolizumab", phases=["PHASE1"])]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        zero = next(
            b for b in result.buckets if b.group == "nivolumab" and b.label == "Phase 1"
        )
        assert zero.value == 0
        assert zero.nct_ids == []

    def test_every_label_is_emitted_once_per_group(self):
        studies = [
            _study("NCT1", "pembrolizumab", phases=["PHASE1"]),
            _study("NCT2", "nivolumab", phases=["PHASE3"]),
        ]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        assert len(result.buckets) == 2 * 2  # {Phase 1, Phase 3} x 2 groups
        assert [b.group for b in result.buckets[:2]] == GROUPS

    def test_group_identity_is_preserved_on_every_bucket(self):
        studies = [_study("NCT1", "pembrolizumab", phases=["PHASE2"])]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        assert all(b.group in GROUPS for b in result.buckets)

    def test_citations_retained_per_group(self):
        # The IDs behind each series must stay separated, or a nivolumab bar
        # could cite a pembrolizumab trial.
        studies = [
            _study("NCT1", "pembrolizumab", phases=["PHASE3"]),
            _study("NCT2", "nivolumab", phases=["PHASE3"]),
        ]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        by_group = {b.group: b.nct_ids for b in result.buckets if b.label == "Phase 3"}
        assert by_group["pembrolizumab"] == ["NCT1"]
        assert by_group["nivolumab"] == ["NCT2"]

    def test_country_dimension_is_supported(self):
        studies = [
            _study("NCT1", "pembrolizumab", countries=["France", "Japan"]),
            _study("NCT2", "nivolumab", countries=["France"]),
        ]
        result = comparison(studies, Dimension.COUNTRY, groups=GROUPS)
        counts = {(b.label, b.group): b.value for b in result.buckets}
        assert counts[("France", "pembrolizumab")] == 1
        assert counts[("France", "nivolumab")] == 1
        assert counts[("Japan", "nivolumab")] == 0

    def test_label_order_follows_combined_totals(self):
        # Both series must share one axis order, derived from the totals rather
        # than from whichever group happens to be first.
        studies = [
            _study("NCT1", "pembrolizumab", countries=["Japan"]),
            _study("NCT2", "nivolumab", countries=["France"]),
            _study("NCT3", "nivolumab", countries=["France"]),
        ]
        result = comparison(studies, Dimension.COUNTRY, groups=GROUPS)
        assert [b.label for b in result.buckets[:2]] == ["France", "France"]

    def test_overlap_is_disclosed(self):
        # A trial studying both drugs is legitimately in both series; the note
        # is what stops the reader summing the bars into a bogus total.
        studies = [_study("NCT1", "pembrolizumab", phases=["PHASE3"])]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        assert any("counted once in each" in note for note in result.notes)

    def test_empty_group_is_disclosed_not_dropped(self):
        studies = [_study("NCT1", "pembrolizumab", phases=["PHASE3"])]
        result = comparison(studies, Dimension.PHASE, groups=GROUPS)
        assert any("nivolumab" in note for note in result.notes)


class TestComparisonValidation:
    def _plan(self, **overrides) -> QueryPlan:
        base = {
            "intent": "Compare two drugs by phase.",
            "operation": "comparison",
            "dimension": "phase",
            "filters": {},
            "comparison_groups": GROUPS,
            "comparison_field": "drug",
            "title": "A vs B",
        }
        base.update(overrides)
        return QueryPlan.model_validate(base)

    def test_fewer_than_two_groups_is_rejected(self):
        with pytest.raises(ValidationError, match="at least two"):
            self._plan(comparison_groups=["pembrolizumab"])

    def test_duplicate_groups_collapse_and_then_fail_validation(self):
        # "Compare pembrolizumab vs Pembrolizumab" is one group, not two.
        with pytest.raises(ValidationError, match="at least two"):
            self._plan(comparison_groups=["pembrolizumab", "Pembrolizumab"])

    def test_group_names_are_sanitised_like_filters(self):
        plan = self._plan(comparison_groups=['"pembrolizumab",', " nivolumab "])
        assert plan.comparison_groups == ["pembrolizumab", "nivolumab"]

    def test_pinning_the_compared_field_as_a_filter_is_incoherent(self):
        plan = self._plan(filters={"drug": "pembrolizumab"})
        with pytest.raises(PlanValidationError, match="the groups supply that value"):
            validate_plan(plan)

    def test_shared_filters_on_other_fields_are_fine(self):
        validate_plan(self._plan(filters={"status": "RECRUITING"}))


# --------------------------------------------------------------------------
# HTTP layer: one stubbed upstream response per comparison group.
# --------------------------------------------------------------------------


def _raw(nct: str, phases: list[str], sponsor: str = "Acme") -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2021-03-01"},
            },
            "designModule": {"phases": phases, "studyType": "INTERVENTIONAL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor, "class": "INDUSTRY"}
            },
            "contactsLocationsModule": {"locations": [{"country": "France"}]},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "Drug X"}]
            },
        }
    }


COMPARISON_QUERY = "Compare trial phases for pembrolizumab vs nivolumab"


def _comparison_client(monkeypatch, handler):
    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return TestClient(app_module.app)


@pytest.fixture
def comparison_plan(monkeypatch):
    """Pin the comparison plan.

    The deterministic planner does not emit comparisons and the LLM is never
    called in tests, so the plan is supplied directly -- what is under test here
    is the fan-out and response assembly, not intent interpretation.
    """
    plan = QueryPlan.model_validate(
        {
            "intent": "Compare pembrolizumab and nivolumab trials by phase.",
            "operation": "comparison",
            "dimension": "phase",
            "filters": {},
            "comparison_groups": GROUPS,
            "comparison_field": "drug",
            "title": "Pembrolizumab vs Nivolumab by Phase",
        }
    )

    async def fake_plan_query(query, hints=None, llm=None):
        # Hints are merged exactly as the real orchestrator does, so the
        # shared-filter test exercises the real precedence rule.
        merged = plan.model_copy(update={"filters": apply_hints(plan.filters, hints)})
        return PlanningOutcome(plan=merged, planner="stub")

    monkeypatch.setattr(app_module, "plan_query", fake_plan_query)
    return plan


def _group_of(request: httpx.Request) -> str:
    """Which comparison group this upstream request is for."""
    expression = request.url.params.get("filter.advanced", "")
    return "nivolumab" if "nivolumab" in expression else "pembrolizumab"


class TestComparisonOverHTTP:
    def test_each_group_gets_its_own_upstream_query(
        self, monkeypatch, comparison_plan
    ):
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.params.get("filter.advanced", ""))
            return httpx.Response(
                200, json={"totalCount": 1, "studies": [_raw("NCT1", ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            assert client.post("/query", json={"query": COMPARISON_QUERY}).status_code == 200

        assert len(seen) == 2
        assert any('AREA[InterventionName]"pembrolizumab"' in e for e in seen)
        assert any('AREA[InterventionName]"nivolumab"' in e for e in seen)

    def test_shared_filters_apply_to_every_group(self, monkeypatch, comparison_plan):
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.params.get("filter.overallStatus", ""))
            return httpx.Response(
                200, json={"totalCount": 1, "studies": [_raw("NCT1", ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            client.post(
                "/query",
                json={"query": COMPARISON_QUERY, "hints": {"status": "RECRUITING"}},
            )

        assert seen == ["RECRUITING", "RECRUITING"]

    def test_grouped_output_is_renderable(self, monkeypatch, comparison_plan):
        def handler(request):
            group = _group_of(request)
            studies = (
                [_raw("NCT1", ["PHASE3"]), _raw("NCT2", ["PHASE3"])]
                if group == "pembrolizumab"
                else [_raw("NCT9", ["PHASE1"])]
            )
            return httpx.Response(
                200, json={"totalCount": len(studies), "studies": studies}
            )

        with _comparison_client(monkeypatch, handler) as client:
            body = client.post("/query", json={"query": COMPARISON_QUERY}).json()

        viz = body["visualization"]
        assert viz["type"] == "grouped_bar_chart"
        series_field = viz["encoding"]["series"]["field"]
        assert series_field == "group"
        # Every row must be addressable through the declared encoding.
        for row in viz["data"]:
            assert viz["encoding"]["x"]["field"] in row
            assert viz["encoding"]["y"]["field"] in row
            assert row[series_field] in GROUPS

        counts = {(r["phase"], r["group"]): r["trial_count"] for r in viz["data"]}
        assert counts[("Phase 3", "pembrolizumab")] == 2
        assert counts[("Phase 3", "nivolumab")] == 0
        assert counts[("Phase 1", "nivolumab")] == 1

    def test_citations_stay_with_their_series(self, monkeypatch, comparison_plan):
        def handler(request):
            group = _group_of(request)
            nct = "NCT1" if group == "pembrolizumab" else "NCT9"
            return httpx.Response(
                200, json={"totalCount": 1, "studies": [_raw(nct, ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            body = client.post("/query", json={"query": COMPARISON_QUERY}).json()

        rows = {r["group"]: r for r in body["visualization"]["data"] if r["phase"] == "Phase 3"}
        assert [c["nct_id"] for c in rows["pembrolizumab"]["citations"]] == ["NCT1"]
        assert [c["nct_id"] for c in rows["nivolumab"]["citations"]] == ["NCT9"]

    def test_per_group_truncation_is_surfaced(self, monkeypatch, comparison_plan):
        def handler(request):
            group = _group_of(request)
            # pembrolizumab is capped upstream; nivolumab is complete.
            total = 5000 if group == "pembrolizumab" else 1
            return httpx.Response(
                200, json={"totalCount": total, "studies": [_raw("NCT1", ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            meta = client.post("/query", json={"query": COMPARISON_QUERY}).json()["meta"]

        groups = {g["group"]: g for g in meta["groups"]}
        assert groups["pembrolizumab"]["truncated"] is True
        assert groups["pembrolizumab"]["total_available"] == 5000
        assert groups["nivolumab"]["truncated"] is False
        assert meta["truncated"] is True
        assert any("pembrolizumab" in note for note in meta["notes"])

    def test_totals_are_not_silently_combined_across_groups(
        self, monkeypatch, comparison_plan
    ):
        # Group match sets can overlap, so a summed total would be meaningless.
        def handler(request):
            return httpx.Response(
                200, json={"totalCount": 7, "studies": [_raw("NCT1", ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            meta = client.post("/query", json={"query": COMPARISON_QUERY}).json()["meta"]

        # Omitted entirely (the response drops nulls) rather than being a sum.
        assert "total_available" not in meta
        assert [g["total_available"] for g in meta["groups"]] == [7, 7]

    def test_one_failing_group_aborts_the_whole_comparison(
        self, monkeypatch, comparison_plan
    ):
        # A chart missing one series reads as a finding, not as an outage.
        def handler(request):
            if _group_of(request) == "nivolumab":
                return httpx.Response(400, text="bad filter expression")
            return httpx.Response(
                200, json={"totalCount": 1, "studies": [_raw("NCT1", ["PHASE3"])]}
            )

        with _comparison_client(monkeypatch, handler) as client:
            response = client.post("/query", json={"query": COMPARISON_QUERY})

        assert response.status_code == 502
        error = response.json()["error"]
        assert error["code"] == "upstream_error"
        assert error["details"]["failed_group"] == "nivolumab"
