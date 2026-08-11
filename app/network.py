"""Sponsor-drug relationship graph. Deterministic, like every other analysis.

This is the one analysis whose result is not a list of buckets, so it gets its
own module and its own result type rather than being bent into the `Bucket`
shape. The LLM chooses that a network was asked for; it never constructs one.

The whole graph is derived from two fields of each study -- the lead sponsor and
its DRUG interventions -- and every edge carries the NCT IDs of the trials that
created it, so any relationship shown can be checked against the registry.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from app.schemas.study import Study

# Only therapeutic agents become nodes. Studies also register procedures,
# devices, behavioural arms, and diagnostics; including those would turn a
# sponsor-drug map into a sponsor-everything map.
#
# BIOLOGICAL is included because the registry uses it interchangeably with DRUG
# for the same molecule: across a 500-study melanoma sample, pembrolizumab was
# typed DRUG 29 times and BIOLOGICAL 18 times. Accepting only DRUG would split
# one node in two and silently drop roughly a third of the edges for exactly the
# monoclonal antibodies these questions are usually about. Narrow this set to
# {"DRUG"} for a stricter reading; the choice is disclosed in the response notes.
DRUG_INTERVENTION_TYPES = {"DRUG", "BIOLOGICAL"}

SPONSOR_NODE = "sponsor"
DRUG_NODE = "drug"


@dataclass
class Node:
    id: str
    label: str
    node_type: str  # "sponsor" | "drug"
    # Unique trials this entity appears in, across the whole graph -- including
    # relationships later removed by the edge cap.
    trial_count: int = 0


@dataclass
class Edge:
    source: str  # sponsor node id
    target: str  # drug node id
    trial_count: int
    nct_ids: list[str] = field(default_factory=list)


@dataclass
class NetworkResult:
    nodes: list[Node]
    edges: list[Edge]
    notes: list[str] = field(default_factory=list)


def _canonical(name: str) -> str:
    """Collapse whitespace so 'Merck  Sharp &\tDohme' matches 'Merck Sharp & Dohme'."""
    return " ".join(name.split())


def build_network(studies: list[Study], top_edges: int | None = None) -> NetworkResult:
    """Build the sponsor-drug graph from normalized studies.

    Edge weight is the number of *distinct trials* linking a sponsor to a drug.
    A trial contributes at most once to any given edge no matter how many times
    it repeats an intervention name, and entity identity is case-insensitive so
    "Pembrolizumab" and "pembrolizumab" are one node.
    """
    # key -> display label, first spelling wins so output is stable given input.
    sponsor_labels: dict[str, str] = {}
    drug_labels: dict[str, str] = {}
    # (sponsor_key, drug_key) -> ordered unique NCT IDs.
    pair_trials: dict[tuple[str, str], list[str]] = defaultdict(list)
    # node key -> unique NCT IDs, for node trial_count.
    sponsor_trials: dict[str, set[str]] = defaultdict(set)
    drug_trials: dict[str, set[str]] = defaultdict(set)

    skipped_no_sponsor = 0
    skipped_no_drug = 0

    for study in studies:
        sponsor = _canonical(study.lead_sponsor or "")
        if not sponsor:
            skipped_no_sponsor += 1
            continue

        # Dedupe interventions within the study: a trial listing the same drug
        # in three arms is still one trial for this relationship.
        drugs: dict[str, str] = {}
        for intervention in study.interventions:
            if (intervention.type or "").upper() not in DRUG_INTERVENTION_TYPES:
                continue
            name = _canonical(intervention.name)
            if name:
                drugs.setdefault(name.casefold(), name)

        if not drugs:
            skipped_no_drug += 1
            continue

        sponsor_key = sponsor.casefold()
        sponsor_labels.setdefault(sponsor_key, sponsor)
        sponsor_trials[sponsor_key].add(study.nct_id)

        for drug_key, drug in drugs.items():
            drug_labels.setdefault(drug_key, drug)
            drug_trials[drug_key].add(study.nct_id)
            trials = pair_trials[(sponsor_key, drug_key)]
            if study.nct_id not in trials:
                trials.append(study.nct_id)

    edges = [
        Edge(
            source=sponsor_labels[sponsor_key],
            target=drug_labels[drug_key],
            trial_count=len(nct_ids),
            nct_ids=nct_ids,
        )
        for (sponsor_key, drug_key), nct_ids in pair_trials.items()
    ]
    # Heaviest first; names break ties so the same input always yields the same
    # graph, which is what makes the cap reproducible.
    edges.sort(key=lambda e: (-e.trial_count, e.source.casefold(), e.target.casefold()))

    notes: list[str] = []
    total_edges = len(edges)
    if top_edges is not None and total_edges > top_edges:
        edges = edges[:top_edges]
        notes.append(
            f"Showing the {top_edges} strongest of {total_edges} sponsor-drug "
            "relationships, ranked by number of shared trials."
        )

    # Only entities that survive the cap are emitted, so the graph has no
    # orphan nodes.
    kept_sponsors = {e.source.casefold() for e in edges}
    kept_drugs = {e.target.casefold() for e in edges}

    nodes = [
        Node(
            id=sponsor_labels[key],
            label=sponsor_labels[key],
            node_type=SPONSOR_NODE,
            trial_count=len(sponsor_trials[key]),
        )
        for key in sorted(kept_sponsors, key=lambda k: -len(sponsor_trials[k]))
    ] + [
        Node(
            id=drug_labels[key],
            label=drug_labels[key],
            node_type=DRUG_NODE,
            trial_count=len(drug_trials[key]),
        )
        for key in sorted(kept_drugs, key=lambda k: -len(drug_trials[k]))
    ]

    notes.append(
        "Nodes are lead sponsors and their drug interventions (registry types "
        f"{'/'.join(sorted(DRUG_INTERVENTION_TYPES))}); an edge means the sponsor "
        "ran at least one trial using that drug, weighted by the number of "
        "distinct trials. Drug names are matched case-insensitively as recorded "
        "by the sponsor, not resolved to a common vocabulary."
    )
    if top_edges is not None and total_edges > top_edges:
        notes.append(
            "Node trial_count reflects the entity across all relationships found, "
            "including those removed by the cap, so node counts exceed the sum of "
            "the edges shown."
        )
    if skipped_no_sponsor:
        notes.append(
            f"{skipped_no_sponsor} study/studies had no lead sponsor recorded and "
            "are excluded from the graph."
        )
    if skipped_no_drug:
        notes.append(
            f"{skipped_no_drug} study/studies listed no drug intervention "
            "(observational, procedure, device, or behavioural) and are excluded."
        )

    return NetworkResult(nodes=nodes, edges=edges, notes=notes)
