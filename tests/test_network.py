"""Sponsor-drug network construction.

The graph is derived entirely from two study fields, so these tests pin the
edge-weight invariant (one trial contributes at most once to an edge) and the
edge cases that would quietly distort it.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as app_module
from app.network import build_network
from app.planning import PlanningOutcome
from app.schemas.plan import QueryPlan
from app.schemas.study import Intervention, Study


def _study(nct: str, sponsor: str | None, drugs: list[str], **kwargs) -> Study:
    return Study(
        nct_id=nct,
        brief_title=f"Trial {nct}",
        lead_sponsor=sponsor,
        interventions=[Intervention(name=d, type="DRUG") for d in drugs],
        **kwargs,
    )


class TestNetworkConstruction:
    def test_sponsor_drug_network(self):
        result = build_network([_study("NCT1", "Merck", ["Pembrolizumab"])])
        assert {(n.id, n.node_type) for n in result.nodes} == {
            ("Merck", "sponsor"),
            ("Pembrolizumab", "drug"),
        }
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert (edge.source, edge.target, edge.trial_count) == (
            "Merck",
            "Pembrolizumab",
            1,
        )

    def test_multiple_drugs_in_one_study_make_multiple_edges(self):
        result = build_network([_study("NCT1", "Merck", ["Pembro", "Chemo"])])
        assert {e.target for e in result.edges} == {"Pembro", "Chemo"}
        assert all(e.trial_count == 1 for e in result.edges)

    def test_duplicate_interventions_do_not_double_count(self):
        # The same drug named across three arms is still one trial.
        result = build_network([_study("NCT1", "Merck", ["Pembro", "Pembro", "Pembro"])])
        assert len(result.edges) == 1
        assert result.edges[0].trial_count == 1
        assert result.edges[0].nct_ids == ["NCT1"]

    def test_repeated_pair_across_trials_increments_the_weight(self):
        result = build_network(
            [
                _study("NCT1", "Merck", ["Pembro"]),
                _study("NCT2", "Merck", ["Pembro"]),
                _study("NCT3", "Merck", ["Pembro"]),
            ]
        )
        assert result.edges[0].trial_count == 3
        assert result.edges[0].nct_ids == ["NCT1", "NCT2", "NCT3"]

    def test_case_and_whitespace_differences_collapse_to_one_node(self):
        result = build_network(
            [
                _study("NCT1", "Merck  Sharp & Dohme", ["Pembrolizumab"]),
                _study("NCT2", "merck sharp & dohme", ["PEMBROLIZUMAB"]),
            ]
        )
        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.edges[0].trial_count == 2
        # First spelling seen wins, so output is stable for a given input.
        assert result.edges[0].source == "Merck Sharp & Dohme"

    def test_edge_weight_counts_trials_not_intervention_rows(self):
        # The invariant the whole graph rests on.
        result = build_network(
            [
                _study("NCT1", "Merck", ["Pembro", "Pembro"]),
                _study("NCT2", "Merck", ["Pembro"]),
            ]
        )
        edge = result.edges[0]
        assert edge.trial_count == len(edge.nct_ids) == len(set(edge.nct_ids)) == 2


class TestNetworkEdgeCases:
    def test_missing_sponsor_is_excluded_and_disclosed(self):
        result = build_network(
            [
                _study("NCT1", None, ["Pembro"]),
                _study("NCT2", "Merck", ["Pembro"]),
            ]
        )
        assert [e.trial_count for e in result.edges] == [1]
        assert any("no lead sponsor" in note for note in result.notes)

    def test_blank_sponsor_is_treated_as_missing(self):
        result = build_network([_study("NCT1", "   ", ["Pembro"])])
        assert result.edges == []

    def test_study_with_no_drug_interventions_is_excluded(self):
        no_drugs = Study(
            nct_id="NCT1",
            lead_sponsor="Merck",
            interventions=[Intervention(name="Counselling", type="BEHAVIORAL")],
        )
        result = build_network([no_drugs, _study("NCT2", "Merck", ["Pembro"])])
        assert [(e.source, e.target) for e in result.edges] == [("Merck", "Pembro")]
        assert any("no drug intervention" in note for note in result.notes)

    def test_biologicals_count_as_drugs(self):
        # The registry types the same monoclonal antibody as DRUG in one study
        # and BIOLOGICAL in the next; treating them differently would split one
        # node in two and drop real edges.
        studies = [
            Study(
                nct_id="NCT1",
                lead_sponsor="Merck",
                interventions=[Intervention(name="Pembrolizumab", type="DRUG")],
            ),
            Study(
                nct_id="NCT2",
                lead_sponsor="Merck",
                interventions=[Intervention(name="Pembrolizumab", type="BIOLOGICAL")],
            ),
        ]
        result = build_network(studies)
        assert len(result.edges) == 1
        assert result.edges[0].trial_count == 2

    def test_non_therapeutic_intervention_types_are_excluded(self):
        for kind in ("PROCEDURE", "DEVICE", "BEHAVIORAL", "DIAGNOSTIC_TEST", "OTHER"):
            study = Study(
                nct_id="NCT1",
                lead_sponsor="Merck",
                interventions=[Intervention(name="Thing", type=kind)],
            )
            assert build_network([study]).edges == [], kind

    def test_untyped_interventions_are_not_assumed_to_be_drugs(self):
        untyped = Study(
            nct_id="NCT1",
            lead_sponsor="Merck",
            interventions=[Intervention(name="Something", type=None)],
        )
        assert build_network([untyped]).edges == []

    def test_observational_study_produces_no_edges(self):
        result = build_network([Study(nct_id="NCT1", lead_sponsor="NIH")])
        assert result.nodes == []
        assert result.edges == []


class TestTopNCap:
    def _wide_graph(self) -> list[Study]:
        # Sponsor i runs drug i in (i + 1) trials, so weights are all distinct.
        return [
            _study(f"NCT{i}_{n}", f"Sponsor {i}", [f"Drug {i}"])
            for i in range(10)
            for n in range(i + 1)
        ]

    def test_cap_keeps_the_heaviest_edges(self):
        result = build_network(self._wide_graph(), top_edges=3)
        assert len(result.edges) == 3
        assert [e.trial_count for e in result.edges] == [10, 9, 8]

    def test_cap_is_disclosed(self):
        result = build_network(self._wide_graph(), top_edges=3)
        assert any("strongest of 10" in note for note in result.notes)

    def test_cap_is_deterministic_under_ties(self):
        # Equal weights must break on name, not on dict ordering.
        studies = [
            _study("NCT1", "Zeta", ["Drug"]),
            _study("NCT2", "Alpha", ["Drug"]),
            _study("NCT3", "Mid", ["Drug"]),
        ]
        first = build_network(studies, top_edges=2)
        second = build_network(list(reversed(studies)), top_edges=2)
        assert [e.source for e in first.edges] == ["Alpha", "Mid"]
        assert [e.source for e in first.edges] == [e.source for e in second.edges]

    def test_cap_leaves_no_orphan_nodes(self):
        result = build_network(self._wide_graph(), top_edges=3)
        referenced = {e.source for e in result.edges} | {e.target for e in result.edges}
        assert {n.id for n in result.nodes} == referenced

    def test_node_counts_span_the_uncapped_graph_and_say_so(self):
        result = build_network(self._wide_graph(), top_edges=3)
        assert any("removed by the cap" in note for note in result.notes)

    def test_no_cap_note_when_nothing_is_dropped(self):
        result = build_network([_study("NCT1", "Merck", ["Pembro"])], top_edges=10)
        assert not any("strongest of" in note for note in result.notes)


class TestNodeCounts:
    def test_node_trial_count_is_unique_trials_for_that_entity(self):
        result = build_network(
            [
                _study("NCT1", "Merck", ["Pembro", "Chemo"]),
                _study("NCT2", "Merck", ["Pembro"]),
                _study("NCT3", "Pfizer", ["Pembro"]),
            ]
        )
        nodes = {n.id: n.trial_count for n in result.nodes}
        assert nodes["Merck"] == 2  # NCT1, NCT2 -- not 3 intervention rows
        assert nodes["Pembro"] == 3
        assert nodes["Chemo"] == 1


# --------------------------------------------------------------------------
# HTTP layer, upstream stubbed.
# --------------------------------------------------------------------------


def _raw(nct: str, sponsor: str, interventions: list[tuple[str, str]]) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2021-03-01"},
            },
            "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor, "class": "INDUSTRY"}
            },
            "contactsLocationsModule": {"locations": [{"country": "France"}]},
            "armsInterventionsModule": {
                "interventions": [{"type": t, "name": n} for n, t in interventions]
            },
        }
    }


NETWORK_QUERY = "Show a network of sponsors and drugs for melanoma trials"

FIXTURE = {
    "totalCount": 3,
    "studies": [
        _raw("NCT1", "Merck", [("Pembrolizumab", "DRUG"), ("Pembrolizumab", "DRUG")]),
        _raw("NCT2", "Merck", [("Pembrolizumab", "DRUG")]),
        _raw("NCT3", "BMS", [("Nivolumab", "DRUG"), ("Counselling", "BEHAVIORAL")]),
    ],
}


@pytest.fixture
def network_client(monkeypatch):
    """TestClient with a pinned network plan and a stubbed upstream."""
    plan = QueryPlan.model_validate(
        {
            "intent": "Map sponsors to the drugs they study in melanoma trials.",
            "operation": "network",
            "dimension": "sponsor",
            "filters": {"condition": "melanoma"},
            "comparison_groups": [],
            "title": "Melanoma Sponsor-Drug Network",
        }
    )

    async def fake_plan_query(query, hints=None, llm=None):
        return PlanningOutcome(plan=plan, planner="stub")

    monkeypatch.setattr(app_module, "plan_query", fake_plan_query)

    def handler(request):
        return httpx.Response(200, json=FIXTURE)

    original_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    with TestClient(app_module.app) as client:
        yield client


class TestNetworkOverHTTP:
    def test_returns_nodes_and_edges_not_chart_rows(self, network_client):
        viz = network_client.post("/query", json={"query": NETWORK_QUERY}).json()[
            "visualization"
        ]
        assert viz["type"] == "network_graph"
        assert viz["data"] == []
        assert {n["id"] for n in viz["nodes"]} == {
            "Merck",
            "BMS",
            "Pembrolizumab",
            "Nivolumab",
        }
        assert len(viz["edges"]) == 2

    def test_edge_weights_survive_the_full_pipeline(self, network_client):
        viz = network_client.post("/query", json={"query": NETWORK_QUERY}).json()[
            "visualization"
        ]
        weights = {(e["source"], e["target"]): e["trial_count"] for e in viz["edges"]}
        # NCT1 lists Pembrolizumab twice; it still counts once.
        assert weights[("Merck", "Pembrolizumab")] == 2
        assert weights[("BMS", "Nivolumab")] == 1

    def test_edge_citations_match_the_contributing_trials(self, network_client):
        viz = network_client.post("/query", json={"query": NETWORK_QUERY}).json()[
            "visualization"
        ]
        edges = {(e["source"], e["target"]): e for e in viz["edges"]}
        merck = edges[("Merck", "Pembrolizumab")]
        assert {c["nct_id"] for c in merck["citations"]} == {"NCT1", "NCT2"}
        # The excerpt must evidence both halves of the relationship.
        excerpt = merck["citations"][0]["excerpt"]
        assert "Merck" in excerpt and "Pembrolizumab" in excerpt
        assert merck["citations"][0]["url"].endswith(merck["citations"][0]["nct_id"])

    def test_every_edge_carries_citations(self, network_client):
        viz = network_client.post("/query", json={"query": NETWORK_QUERY}).json()[
            "visualization"
        ]
        assert all(edge["citations"] for edge in viz["edges"])

    def test_meta_discloses_how_the_graph_was_built(self, network_client):
        meta = network_client.post("/query", json={"query": NETWORK_QUERY}).json()["meta"]
        assert meta["query_interpretation"]["operation"] == "network"
        assert any("Nodes are lead sponsors" in note for note in meta["notes"])

    def test_no_extractable_relationships_returns_404(self, monkeypatch):
        # Understood, matched studies, but nothing to draw -- not a 500.
        plan = QueryPlan.model_validate(
            {
                "intent": "Map sponsors to drugs.",
                "operation": "network",
                "dimension": "sponsor",
                "filters": {"condition": "melanoma"},
                "comparison_groups": [],
                "title": "Network",
            }
        )

        async def fake_plan_query(query, hints=None, llm=None):
            return PlanningOutcome(plan=plan, planner="stub")

        monkeypatch.setattr(app_module, "plan_query", fake_plan_query)

        observational = {
            "totalCount": 1,
            "studies": [_raw("NCT1", "NIH", [("Survey", "BEHAVIORAL")])],
        }

        def handler(request):
            return httpx.Response(200, json=observational)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
        with TestClient(app_module.app) as client:
            response = client.post("/query", json={"query": NETWORK_QUERY})

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "no_results"
