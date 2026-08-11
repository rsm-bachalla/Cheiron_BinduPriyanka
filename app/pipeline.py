"""Orchestration: plan -> fetch -> normalize -> aggregate -> visualize -> cite.

Deliberately a straight line. Each stage is an ordinary function over typed
inputs, so any stage can be tested without the others and the whole flow can be
read top to bottom.
"""

import logging

from app.aggregate import aggregate
from app.citations import build_citations
from app.clinicaltrials import ClinicalTrialsClient
from app.config import Settings
from app.errors import NoResultsError, UnsupportedAnalysisError
from app.plan_validation import IMPLEMENTED_OPERATIONS
from app.normalize import normalize_studies
from app.schemas.api import (
    Meta,
    QueryHints,
    QueryInterpretation,
    QueryResponse,
)
from app.schemas.plan import QueryPlan
from app.viz import build_visualization

logger = logging.getLogger(__name__)


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

    raw_studies, total_available = await client.search(plan.filters)

    if not raw_studies:
        raise NoResultsError(
            "No trials on ClinicalTrials.gov matched these filters.",
            filters=plan.filters.describe(),
        )

    studies = normalize_studies(raw_studies)
    if not studies:
        raise NoResultsError(
            "Matching records could not be interpreted.",
            filters=plan.filters.describe(),
        )

    result = aggregate(plan.operation, studies, plan.dimension)

    # Bucket -> citations. The IDs already live on the bucket; this only renders.
    studies_by_id = {s.nct_id: s for s in studies}
    citations_by_label = {
        bucket.label: build_citations(
            bucket, studies_by_id, plan.dimension, settings.citations_per_point
        )
        for bucket in result.buckets
    }

    visualization = build_visualization(
        operation=plan.operation,
        dimension=plan.dimension,
        title=plan.title,
        buckets=result.buckets,
        citations_by_label=citations_by_label,
    )

    truncated = total_available is not None and len(studies) < total_available
    notes = list(result.notes)
    if truncated:
        notes.insert(
            0,
            f"Analysed {len(studies)} of {total_available} matching trials "
            f"(capped at {settings.ctgov_max_records}); counts are a sample, "
            "not registry totals.",
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
            notes=notes,
        ),
    )
