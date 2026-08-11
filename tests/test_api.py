"""End-to-end tests through the HTTP layer, with the upstream API stubbed.

No network access, so the suite is deterministic and runs offline.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


def _study(nct: str, phases: list[str], title: str = "A trial") -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": title},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2021-03-01"},
            },
            "designModule": {"phases": phases, "studyType": "INTERVENTIONAL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Acme", "class": "INDUSTRY"}
            },
            "contactsLocationsModule": {"locations": [{"country": "France"}]},
            "armsInterventionsModule": {"interventions": [{"name": "Drug X"}]},
        }
    }


FIXTURE = {
    "totalCount": 4,
    "studies": [
        _study("NCT0000001", ["PHASE3"], "Trial A"),
        _study("NCT0000002", ["PHASE3"], "Trial B"),
        _study("NCT0000003", ["PHASE1", "PHASE2"], "Trial C"),
        _study("NCT0000004", [], "Observational D"),
    ],
}


@pytest.fixture
def client(monkeypatch):
    """TestClient whose upstream calls are served from the fixture above."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=FIXTURE)

    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}


class TestQuerySuccess:
    def test_returns_renderable_visualization(self, client):
        response = client.post(
            "/query",
            json={"query": "How are breast cancer trials distributed across phases?"},
        )
        assert response.status_code == 200
        viz = response.json()["visualization"]
        assert viz["type"] == "bar_chart"
        # Every row must be addressable through the declared encoding, or a
        # frontend cannot render it generically.
        x_field = viz["encoding"]["x"]["field"]
        y_field = viz["encoding"]["y"]["field"]
        assert all(x_field in row and y_field in row for row in viz["data"])

    def test_counts_reconcile_with_record_count(self, client):
        body = client.post(
            "/query", json={"query": "breast cancer trials by phase"}
        ).json()
        total = sum(row["trial_count"] for row in body["visualization"]["data"])
        assert total == body["meta"]["record_count"] == 4

    def test_every_data_point_carries_citations(self, client):
        body = client.post(
            "/query", json={"query": "breast cancer trials by phase"}
        ).json()
        for row in body["visualization"]["data"]:
            assert row["citations"], f"no citations for {row}"
            for citation in row["citations"]:
                assert citation["nct_id"].startswith("NCT")
                assert citation["url"].endswith(citation["nct_id"])

    def test_citations_come_from_the_bucket_they_annotate(self, client):
        # Guards the traceability claim: a Phase 3 bar must cite Phase 3 trials.
        body = client.post(
            "/query", json={"query": "breast cancer trials by phase"}
        ).json()
        rows = {row["phase"]: row for row in body["visualization"]["data"]}
        assert {c["nct_id"] for c in rows["Phase 3"]["citations"]} == {
            "NCT0000001",
            "NCT0000002",
        }
        assert rows["Phase 1/Phase 2"]["citations"][0]["excerpt"] == "Phase: PHASE1, PHASE2"

    def test_meta_reports_interpretation_and_provenance(self, client):
        meta = client.post(
            "/query", json={"query": "breast cancer trials by phase"}
        ).json()["meta"]
        assert meta["source"] == "ClinicalTrials.gov"
        assert meta["query_interpretation"]["planner"] == "rulebased"
        assert meta["filters"]["condition"] == "breast cancer"

    def test_hints_are_applied(self, client):
        meta = client.post(
            "/query",
            json={
                "query": "How are breast cancer trials distributed across phases?",
                "hints": {"country": "France"},
            },
        ).json()["meta"]
        assert meta["filters"]["country"] == "France"


class TestQueryErrors:
    def test_ambiguous_query_returns_structured_422(self, client):
        response = client.post("/query", json={"query": "tell me about cancer"})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "unsupported_query"
        assert error["details"]["reason"]

    def test_invalid_payload_rejected(self, client):
        assert client.post("/query", json={"query": "ab"}).status_code == 422
        assert client.post("/query", json={}).status_code == 422


class TestNoResults:
    def test_empty_upstream_result_returns_404(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"totalCount": 0, "studies": []})

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
        with TestClient(app) as test_client:
            response = test_client.post(
                "/query", json={"query": "breast cancer trials by phase"}
            )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "no_results"
