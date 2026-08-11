"""Orchestration: plan -> fetch -> normalize -> aggregate -> visualize -> cite.

Deliberately a straight line. Each stage is an ordinary function over typed
inputs, so any stage can be tested without the others and the whole flow can be
read top to bottom.

Two operations bend the line slightly and say so explicitly:
  * comparison fetches once per group, concurrently, and reports per-group meta;
  * network produces nodes and edges instead of chart rows.
"""

import asyncio
import logging

from app.aggregate import aggregate
from app.citations import build_citations, build_edge_citations
from app.clinicaltrials import ClinicalTrialsClient
from app.config import Settings
from app.errors import NoResultsError, UnsupportedAnalysisError, UpstreamError
from app.network import build_network
from app.plan_validation import IMPLEMENTED_OPERATIONS
from app.normalize import normalize_studies
from app.schemas.api import (
    GroupMeta,
    Meta,
    QueryHints,
    QueryInterpretation,
    QueryResponse,
    Visualization,
)
from app.schemas.plan import AnalysisOp, QueryPlan, TrialFilters
from app.schemas.study import Study
from app.viz import build_network_visualization, build_visualization

logger = logging.getLogger(__name__)


async def _fetch_group(
    client: ClinicalTrialsClient, plan: QueryPlan, group: str
) -> tuple[list[Study], int | None]:
    """Run one comparison group's own upstream query.

    The group value replaces exactly one filter field -- the one the planner
    named in `comparison_field` -- so every shared filter in the plan still
    applies to each group. Membership is therefore decided by ClinicalTrials.gov
    matching, never by guessing locally which studies belong to which group.
    """
    merged = plan.filters.model_dump()
    merged[plan.comparison_field.value] = group
    # Rebuilt through the model rather than model_copy so the group name goes
    # through the same sanitisation as any other filter value.
    filters = TrialFilters(**merged)

    raw_studies, total = await client.search(filters)
    return normalize_studies(raw_studies, group=group), total


async def _fetch_comparison(
    client: ClinicalTrialsClient, plan: QueryPlan, settings: Settings
) -> tuple[list[Study], list[GroupMeta], list[str]]:
    """Fetch every comparison group concurrently.

    The groups are independent queries, so they run in parallel: the request
    costs one round trip rather than N. If any group fails, the whole comparison
    fails -- a chart missing one of its two series looks like a real finding
    ("nivolumab has no phase 3 trials") when it is actually an outage.
    """
    groups = plan.comparison_groups
    results = await asyncio.gather(
        *(_fetch_group(client, plan, group) for group in groups),
        return_exceptions=True,
    )

    studies: list[Study] = []
    group_meta: list[GroupMeta] = []
    notes: list[str] = []

    for group, result in zip(groups, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Comparison group %r failed upstream: %s", group, result)
            raise UpstreamError(
                "Comparison aborted: one group could not be retrieved, and a "
                "partial comparison would be misleading.",
                failed_group=group,
                groups=groups,
                cause=str(result),
            )

        group_studies, total = result
        studies.extend(group_studies)

        truncated = total is not None and len(group_studies) < total
        group_meta.append(
            GroupMeta(
                group=group,
                record_count=len(group_studies),
                total_available=total,
                truncated=truncated,
            )
        )
        if truncated:
            notes.append(
                f"'{group}': analysed {len(group_studies)} of {total} matching "
                f"trials (capped at {settings.ctgov_max_records}); this series "
                "is a sample, not registry totals."
            )

    return studies, group_meta, notes


def _build_network_response(
    plan: QueryPlan, studies: list[Study], settings: Settings
) -> tuple[Visualization, list[str]]:
    """Aggregate and render the sponsor-drug graph."""
    result = build_network(studies, top_edges=settings.network_top_edges)

    if not result.edges:
        raise NoResultsError(
            "No sponsor-drug relationships could be built from the matching "
            "trials.",
            filters=plan.filters.describe(),
            reason="matching studies had no lead sponsor or no DRUG intervention",
        )

    studies_by_id = {s.nct_id: s for s in studies}
    citations = [
        build_edge_citations(edge, studies_by_id, settings.citations_per_point)
        for edge in result.edges
    ]

    visualization = build_network_visualization(
        title=plan.title,
        nodes=result.nodes,
        edges=result.edges,
        citations=citations,
    )
    return visualization, result.notes


def _build_chart_response(
    plan: QueryPlan, studies: list[Study], settings: Settings
) -> tuple[Visualization, list[str]]:
    """Aggregate and render any of the bucket-shaped analyses."""
    kwargs = (
        {"groups": plan.comparison_groups}
        if plan.operation is AnalysisOp.COMPARISON
        else {}
    )
    result = aggregate(plan.operation, studies, plan.dimension, **kwargs)

    # Bucket -> citations. The IDs already live on the bucket; this only renders.
    # Positional, because a comparison repeats each label once per group.
    studies_by_id = {s.nct_id: s for s in studies}
    citations = [
        build_citations(
            bucket, studies_by_id, plan.dimension, settings.citations_per_point
        )
        for bucket in result.buckets
    ]

    visualization = build_visualization(
        operation=plan.operation,
        dimension=plan.dimension,
        title=plan.title,
        buckets=result.buckets,
        citations=citations,
    )
    return visualization, result.notes


async def run_query(
    *,
    query: str,
    hints: QueryHints | None,
    plan: QueryPlan,
    planner_name: str,
    client: ClinicalTrialsClient,
    settings: Settings,
) -> QueryResponse:
    """Execute a validated plan and build the response."""
    if plan.operation not in IMPLEMENTED_OPERATIONS:
        # The intent was understood; the aggregator simply does not exist yet.
        # Say so precisely rather than failing as an internal error.
        raise UnsupportedAnalysisError(
            f"'{plan.operation.value}' analysis is not implemented yet.",
            operation=plan.operation.value,
            interpreted_intent=plan.intent,
            implemented=sorted(op.value for op in IMPLEMENTED_OPERATIONS),
        )

    fetch_notes: list[str] = []
    group_meta: list[GroupMeta] | None = None
    total_available: int | None = None

    if plan.operation is AnalysisOp.COMPARISON:
        studies, group_meta, fetch_notes = await _fetch_comparison(
            client, plan, settings
        )
        # Left null on purpose: group match sets overlap, so a combined total
        # would not mean anything. Per-group totals live in `meta.groups`.
        truncated = any(g.truncated for g in group_meta)
        empty_reason = "none of the comparison groups returned any trials"
    else:
        raw_studies, total_available = await client.search(plan.filters)
        studies = normalize_studies(raw_studies)
        truncated = total_available is not None and len(studies) < total_available
        empty_reason = "no matching studies"

    if not studies:
        raise NoResultsError(
            "No trials on ClinicalTrials.gov matched these filters.",
            filters=plan.filters.describe(),
            reason=empty_reason,
        )

    if plan.operation is AnalysisOp.NETWORK:
        visualization, analysis_notes = _build_network_response(plan, studies, settings)
    else:
        visualization, analysis_notes = _build_chart_response(plan, studies, settings)

    notes = fetch_notes + analysis_notes
    if truncated and plan.operation is not AnalysisOp.COMPARISON:
        notes.insert(
            0,
            f"Analysed {len(studies)} of {total_available} matching trials "
            f"(capped at {settings.ctgov_max_records}); counts are a sample, "
            "not registry totals.",
        )

    logger.info(
        "Answered via %s: operation=%s dimension=%s records=%d",
        planner_name,
        plan.operation.value,
        plan.dimension.value,
        len(studies),
    )

    return QueryResponse(
        visualization=visualization,
        meta=Meta(
            query_interpretation=QueryInterpretation(
                intent=plan.intent,
                operation=plan.operation.value,
                dimension=plan.dimension.value,
                planner=planner_name,
            ),
            filters=plan.filters.describe(),
            record_count=len(studies),
            total_available=total_available,
            truncated=truncated,
            groups=group_meta,
            notes=notes,
        ),
    )
