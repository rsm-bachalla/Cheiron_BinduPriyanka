"""Client tests: Essie expression construction and pagination behaviour.

The expression assertions encode parameter behaviour verified against the live
API, so a regression here is caught without a network call.
"""

import httpx
import pytest

from app.clinicaltrials import ClinicalTrialsClient, build_filter_expression
from app.config import Settings
from app.errors import UpstreamError
from app.schemas.plan import TrialFilters, TrialPhase, TrialStatus


class TestBuildFilterExpression:
    def test_no_filters_yields_no_expression(self):
        assert build_filter_expression(TrialFilters()) is None

    def test_multi_word_values_are_phrase_quoted(self):
        # Unquoted multi-word values match the words separately, which is a
        # broader and less accurate result set.
        expr = build_filter_expression(TrialFilters(condition="breast cancer"))
        assert expr == 'AREA[ConditionSearch]"breast cancer"'

    def test_uses_intervention_area_not_loose_query(self):
        # Measured on live data: query.intr was 84% precise for pembrolizumab,
        # AREA[InterventionName] was 100%.
        expr = build_filter_expression(TrialFilters(drug="pembrolizumab"))
        assert expr == 'AREA[InterventionName]"pembrolizumab"'

    def test_clauses_combine_with_and(self):
        expr = build_filter_expression(
            TrialFilters(condition="melanoma", phase=TrialPhase.PHASE3)
        )
        assert expr == 'AREA[ConditionSearch]"melanoma" AND AREA[Phase]PHASE3'

    def test_embedded_quotes_are_stripped(self):
        # An unbalanced quote makes the server reject the entire expression
        # with HTTP 400, so values must never be interpolated raw.
        expr = build_filter_expression(TrialFilters(condition='bad"quote'))
        assert expr is not None
        assert expr.count('"') == 2
        assert expr == 'AREA[ConditionSearch]"bad quote"'

    def test_year_range_uses_min_max_sentinels_when_open_ended(self):
        assert "RANGE[2020-01-01,MAX]" in build_filter_expression(
            TrialFilters(start_year=2020)
        )
        assert "RANGE[MIN,2021-12-31]" in build_filter_expression(
            TrialFilters(end_year=2021)
        )


def _client(handler) -> ClinicalTrialsClient:
    transport = httpx.MockTransport(handler)
    return ClinicalTrialsClient(
        httpx.AsyncClient(transport=transport), Settings(ctgov_page_size=2, ctgov_max_records=10)
    )


class TestSearch:
    async def test_status_uses_native_param_not_area_clause(self):
        seen = {}

        def handler(request):
            seen.update(request.url.params)
            return httpx.Response(200, json={"studies": [], "totalCount": 0})

        await _client(handler).search(TrialFilters(status=TrialStatus.RECRUITING))
        assert seen["filter.overallStatus"] == "RECRUITING"
        assert "filter.advanced" not in seen

    async def test_always_requests_total_count(self):
        # totalCount is absent from the payload unless explicitly requested,
        # and truncation disclosure depends on it.
        seen = {}

        def handler(request):
            seen.update(request.url.params)
            return httpx.Response(200, json={"studies": [], "totalCount": 0})

        await _client(handler).search(TrialFilters(condition="x"))
        assert seen["countTotal"] == "true"

    async def test_follows_page_tokens_until_exhausted(self):
        pages = [
            {"studies": [{"a": 1}, {"a": 2}], "totalCount": 3, "nextPageToken": "t1"},
            {"studies": [{"a": 3}]},
        ]
        calls = []

        def handler(request):
            calls.append(request.url.params.get("pageToken"))
            return httpx.Response(200, json=pages[len(calls) - 1])

        studies, total = await _client(handler).search(TrialFilters(condition="x"))
        assert len(studies) == 3
        assert total == 3
        assert calls == [None, "t1"]

    async def test_respects_record_cap(self):
        def handler(request):
            return httpx.Response(
                200,
                json={"studies": [{"a": 1}] * 2, "totalCount": 999, "nextPageToken": "t"},
            )

        client = ClinicalTrialsClient(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            Settings(ctgov_page_size=2, ctgov_max_records=5),
        )
        studies, total = await client.search(TrialFilters(condition="x"))
        assert len(studies) == 5  # capped, not 999
        assert total == 999

    async def test_client_error_surfaces_plain_text_body(self):
        # The API returns plain text, not JSON, on 400.
        def handler(request):
            return httpx.Response(400, text="Invalid value in parameter `overallStatus`")

        with pytest.raises(UpstreamError) as exc:
            await _client(handler).search(TrialFilters(condition="x"))
        assert "overallStatus" in exc.value.details["upstream_message"]

    async def test_does_not_retry_client_errors(self):
        # A bad expression will fail identically every time; retrying only
        # multiplies latency.
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(400, text="bad")

        with pytest.raises(UpstreamError):
            await _client(handler).search(TrialFilters(condition="x"))
        assert len(calls) == 1
