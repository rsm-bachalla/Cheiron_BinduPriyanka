"""Turn a natural-language question into a validated QueryPlan.

This module holds the deterministic pattern planner. It handles a deliberately
narrow set of well-understood phrasings and *refuses* anything else, rather than
guessing an interpretation the caller cannot see. Broad NL coverage is the LLM
planner's job (added in the next step); this remains the zero-credential path and
the fallback when the LLM is unavailable.
"""

import re

from app.errors import UnsupportedQueryError
from app.schemas.api import QueryHints
from app.schemas.plan import (
    AnalysisOp,
    Dimension,
    QueryPlan,
    TrialFilters,
    TrialPhase,
    TrialStatus,
)

# Dimension cues, checked in order. Longer/more specific phrases first so
# "sponsor type" is not swallowed by "sponsor".
_DIMENSION_PATTERNS: list[tuple[Dimension, tuple[str, ...]]] = [
    (Dimension.SPONSOR_TYPE, ("sponsor type", "sponsor class", "industry vs")),
    (Dimension.PHASE, ("phase", "phases")),
    (Dimension.STATUS, ("status", "recruiting status", "recruitment status")),
    (Dimension.COUNTRY, ("country", "countries", "geograph", "region", "worldwide")),
    (Dimension.SPONSOR, ("sponsor", "company", "funder")),
    (Dimension.YEAR, ("over time", "trend", "by year", "per year", "since")),
]

_DISTRIBUTION_CUES = (
    "distribut", "breakdown", "broken down", "across", "by ", "how many", "split",
)

_STATUS_TERMS = {
    "recruiting": TrialStatus.RECRUITING,
    "not yet recruiting": TrialStatus.NOT_YET_RECRUITING,
    "active": TrialStatus.ACTIVE_NOT_RECRUITING,
    "completed": TrialStatus.COMPLETED,
    "terminated": TrialStatus.TERMINATED,
    "withdrawn": TrialStatus.WITHDRAWN,
    "suspended": TrialStatus.SUSPENDED,
}

_PHASE_TERMS = {
    "early phase 1": TrialPhase.EARLY_PHASE1,
    "phase 1": TrialPhase.PHASE1,
    "phase 2": TrialPhase.PHASE2,
    "phase 3": TrialPhase.PHASE3,
    "phase 4": TrialPhase.PHASE4,
}

# Conditions we can recognise without a model. Intentionally small: the point is
# high precision on obvious cases, not coverage.
_CONDITIONS = (
    "breast cancer", "lung cancer", "non-small cell lung cancer", "melanoma",
    "prostate cancer", "colorectal cancer", "pancreatic cancer", "ovarian cancer",
    "leukemia", "lymphoma", "glioblastoma", "multiple myeloma", "diabetes",
    "alzheimer", "parkinson", "asthma", "copd", "obesity", "hypertension",
    "covid-19", "depression", "epilepsy",
)

_SUPPORTED_MESSAGE = (
    "This deterministic planner recognises a narrow set of phrasings "
    "(for example: 'How are breast cancer trials distributed across phases?'). "
    "Enable the LLM planner by setting OPENAI_API_KEY for open-ended questions, "
    "or supply structured hints alongside the query."
)


def _match_dimension(segment: str) -> Dimension | None:
    for dimension, cues in _DIMENSION_PATTERNS:
        if any(cue in segment for cue in cues):
            return dimension
    return None


# The grouping subject follows a marker like "by" / "across" / "per". Anchoring
# on that first matters: in "phase 3 melanoma trials by country", a plain scan
# hits "phase" before "country" and groups by the wrong axis.
_GROUPING_MARKER = re.compile(r"\b(?:by|across|per|between|grouped by)\b(.*)", re.I)


def _detect_dimension(text: str) -> Dimension | None:
    marker = _GROUPING_MARKER.search(text)
    if marker:
        # Prefer whatever the grouping marker actually points at.
        if dimension := _match_dimension(marker.group(1)):
            return dimension
    # No marker (or nothing recognisable after it): fall back to a full scan,
    # which still catches phrasings like "trends over time".
    return _match_dimension(text)


def _detect_condition(text: str) -> str | None:
    # Longest match wins, so "non-small cell lung cancer" beats "lung cancer".
    matches = [c for c in _CONDITIONS if c in text]
    return max(matches, key=len) if matches else None


def _detect_status(text: str) -> TrialStatus | None:
    for term, status in _STATUS_TERMS.items():
        if term in text:
            return status
    return None


def _detect_phase_filter(text: str, dimension: Dimension | None) -> TrialPhase | None:
    # If we are grouping *by* phase, a phase word is the axis, not a filter.
    if dimension is Dimension.PHASE:
        return None
    for term, phase in _PHASE_TERMS.items():
        if term in text:
            return phase
    return None


def _detect_years(text: str) -> tuple[int | None, int | None]:
    years = [int(m.group()) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)]
    if not years:
        return None, None
    if len(years) == 1:
        if "since" in text or "after" in text or "from" in text:
            return years[0], None
        if "before" in text or "until" in text:
            return None, years[0]
        return years[0], years[0]
    return min(years), max(years)


def apply_hints(filters: TrialFilters, hints: QueryHints | None) -> TrialFilters:
    """Overlay caller hints. A set hint always wins over an inferred value."""
    if hints is None:
        return filters
    overrides = {
        "drug": hints.drug_name,
        "condition": hints.condition,
        "phase": hints.trial_phase,
        "sponsor": hints.sponsor,
        "country": hints.country,
        "status": hints.status,
        "start_year": hints.start_year,
        "end_year": hints.end_year,
    }
    merged = filters.model_dump()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return TrialFilters(**merged)


def plan_query(query: str, hints: QueryHints | None = None) -> QueryPlan:
    """Build a plan, or raise UnsupportedQueryError if confidence is too low."""
    text = query.lower().strip()

    dimension = _detect_dimension(text)
    condition = _detect_condition(text)
    status = _detect_status(text)
    start_year, end_year = _detect_years(text)

    filters = TrialFilters(
        condition=condition,
        status=status,
        phase=_detect_phase_filter(text, dimension),
        start_year=start_year,
        end_year=end_year,
    )
    filters = apply_hints(filters, hints)

    # Hints can supply the grouping subject even when the wording is unclear.
    if dimension is None and hints is not None and hints.trial_phase is None:
        dimension = Dimension.PHASE if "phase" in text else dimension

    if dimension is None:
        raise UnsupportedQueryError(
            "Could not determine what to group the trials by.",
            query=query,
            reason="no recognised grouping dimension (phase, status, country, sponsor, year)",
            supported=_SUPPORTED_MESSAGE,
        )

    if not any(cue in text for cue in _DISTRIBUTION_CUES):
        raise UnsupportedQueryError(
            "Could not determine what kind of analysis was requested.",
            query=query,
            reason="no recognised analysis cue (e.g. 'distributed across', 'breakdown by')",
            supported=_SUPPORTED_MESSAGE,
        )

    # An unfiltered query would scan the whole registry and produce a
    # meaningless answer, so require at least one filter.
    if filters.is_empty():
        raise UnsupportedQueryError(
            "No condition, drug, sponsor, or country could be identified.",
            query=query,
            reason="a query with no filters would scan the entire registry",
            supported=_SUPPORTED_MESSAGE,
        )

    subject = filters.condition or filters.drug or filters.sponsor or "Clinical"
    return QueryPlan(
        intent=(
            f"Count {subject} trials grouped by "
            f"{dimension.value.replace('_', ' ')}."
        ),
        operation=AnalysisOp.DISTRIBUTION,
        dimension=dimension,
        filters=filters,
        title=f"{subject.title()} Trials by {dimension.value.replace('_', ' ').title()}",
    )
