"""Semantic validation of a QueryPlan.

Schema validation proves a plan is well-formed; it cannot prove the plan makes
sense. A `time_trend` grouped by `sponsor` satisfies every type constraint and
is still incoherent. These checks are deterministic and live entirely outside
the LLM, so a model change cannot weaken them.
"""

from app.schemas.plan import AnalysisOp, Dimension, QueryPlan

# Which dimensions each operation can meaningfully group by.
ALLOWED_DIMENSIONS: dict[AnalysisOp, set[Dimension]] = {
    # A trend is only a trend along a time axis.
    AnalysisOp.TIME_TREND: {Dimension.YEAR},
    # Geography means location.
    AnalysisOp.GEO: {Dimension.COUNTRY},
    # A relationship graph is anchored on an entity that has relationships.
    AnalysisOp.NETWORK: {Dimension.SPONSOR},
    # Comparisons and plain distributions work across any categorical axis.
    AnalysisOp.COMPARISON: {
        Dimension.PHASE,
        Dimension.STATUS,
        Dimension.YEAR,
        Dimension.COUNTRY,
        Dimension.SPONSOR,
        Dimension.SPONSOR_TYPE,
    },
    AnalysisOp.DISTRIBUTION: {
        Dimension.PHASE,
        Dimension.STATUS,
        Dimension.YEAR,
        Dimension.COUNTRY,
        Dimension.SPONSOR,
        Dimension.SPONSOR_TYPE,
    },
}

# Operations with an aggregator wired up. The planner may legitimately produce
# the others; the pipeline reports them as not-yet-supported rather than failing
# with an unhandled error.
IMPLEMENTED_OPERATIONS = {
    AnalysisOp.DISTRIBUTION,
    AnalysisOp.TIME_TREND,
    AnalysisOp.GEO,
}


class PlanValidationError(Exception):
    """A structurally valid plan that is semantically incoherent."""

    def __init__(self, reason: str, *, repairable: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        # Most incoherence is a prompting problem worth one repair attempt.
        self.repairable = repairable


def validate_plan(plan: QueryPlan) -> None:
    """Raise PlanValidationError if the plan is not coherently executable."""
    allowed = ALLOWED_DIMENSIONS.get(plan.operation, set())
    if plan.dimension not in allowed:
        expected = ", ".join(sorted(d.value for d in allowed))
        raise PlanValidationError(
            f"operation '{plan.operation.value}' cannot group by "
            f"'{plan.dimension.value}'; expected one of: {expected}"
        )

    if plan.operation is AnalysisOp.COMPARISON and len(plan.comparison_groups) < 2:
        raise PlanValidationError(
            "comparison requires at least two comparison_groups"
        )

    if plan.operation is not AnalysisOp.COMPARISON and plan.comparison_groups:
        raise PlanValidationError(
            f"comparison_groups is only valid for 'comparison', not "
            f"'{plan.operation.value}'"
        )

    start, end = plan.filters.start_year, plan.filters.end_year
    if start is not None and end is not None and start > end:
        raise PlanValidationError(
            f"start_year ({start}) is after end_year ({end})"
        )

    # An unfiltered query scans the whole registry and answers nothing useful.
    # Comparisons are exempt: their groups become the filters at fetch time.
    if plan.filters.is_empty() and plan.operation is not AnalysisOp.COMPARISON:
        raise PlanValidationError(
            "no filters were identified; an unfiltered query would scan the "
            "entire registry",
            # Re-prompting cannot invent a subject the question never named.
            repairable=False,
        )
