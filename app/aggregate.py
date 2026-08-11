"""Deterministic aggregation. No LLM involvement at any point.

Every aggregator returns `list[Bucket]`, and each bucket carries the NCT IDs of
the studies that produced it. Citations are therefore a byproduct of counting
rather than a later lookup, which is what makes them impossible to fabricate.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from app.normalize import NO_PHASE_LABEL, phase_label, status_label
from app.schemas.plan import AnalysisOp, Dimension
from app.schemas.study import Study


@dataclass
class Bucket:
    """One aggregated data point, with provenance attached."""

    label: str
    value: int
    nct_ids: list[str] = field(default_factory=list)
    group: str | None = None  # set for grouped/comparison series


@dataclass
class AggregationResult:
    buckets: list[Bucket]
    # Caveats discovered while aggregating (double counting, exclusions, caps).
    # Surfaced verbatim in `meta.notes` so the caller sees how numbers were made.
    notes: list[str] = field(default_factory=list)


# How each dimension turns a study into zero or more bucket labels. Returning a
# list is what lets multi-valued dimensions (country) fan out while single-valued
# ones (phase) stay exact.
def _labels_for(study: Study, dimension: Dimension) -> list[str]:
    match dimension:
        case Dimension.PHASE:
            return [phase_label(study.phases)]
        case Dimension.STATUS:
            return [status_label(study.overall_status)]
        case Dimension.YEAR:
            return [str(study.start_year)] if study.start_year else []
        case Dimension.COUNTRY:
            return list(study.countries)
        case Dimension.SPONSOR:
            return [study.lead_sponsor] if study.lead_sponsor else []
        case Dimension.SPONSOR_TYPE:
            return [study.sponsor_class.title()] if study.sponsor_class else []
    return []


# Phase buckets read most naturally in trial order rather than by count.
_PHASE_ORDER = [
    "Early Phase 1",
    "Phase 1",
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3",
    "Phase 3",
    "Phase 4",
    "Not Applicable",
    NO_PHASE_LABEL,
]


def _sort_buckets(buckets: list[Bucket], dimension: Dimension) -> list[Bucket]:
    """Order buckets by whatever reads correctly for the dimension."""
    if dimension is Dimension.PHASE:
        def phase_key(bucket: Bucket) -> tuple[int, str]:
            try:
                return (_PHASE_ORDER.index(bucket.label), "")
            except ValueError:
                # Unknown combination: keep it stable, after the known ones.
                return (len(_PHASE_ORDER), bucket.label)

        return sorted(buckets, key=phase_key)

    if dimension is Dimension.YEAR:
        return sorted(buckets, key=lambda b: b.label)

    # Everything else is a ranking: descending count, name as a stable tiebreak.
    return sorted(buckets, key=lambda b: (-b.value, b.label))


def distribution(
    studies: list[Study], dimension: Dimension, top_n: int | None = None
) -> AggregationResult:
    """Count studies per distinct value of `dimension`."""
    counts: dict[str, list[str]] = defaultdict(list)
    for study in studies:
        for label in _labels_for(study, dimension):
            counts[label].append(study.nct_id)

    buckets = [
        Bucket(label=label, value=len(ids), nct_ids=ids) for label, ids in counts.items()
    ]
    buckets = _sort_buckets(buckets, dimension)

    notes: list[str] = []

    if dimension is Dimension.PHASE:
        if any(b.label == NO_PHASE_LABEL for b in buckets):
            notes.append(
                f"'{NO_PHASE_LABEL}' covers studies with no phase recorded "
                "(typically observational); it is distinct from 'Not Applicable'."
            )
        if any("/" in b.label for b in buckets):
            notes.append(
                "Studies registered under multiple phases (e.g. 'Phase 1/Phase 2') "
                "form their own bucket and are not counted under each phase, so "
                "bucket counts sum to the study count."
            )

    if dimension is Dimension.COUNTRY:
        notes.append(
            "A study running in several countries is counted once per country, "
            "so the column total exceeds the number of studies analysed."
        )

    if dimension is Dimension.YEAR:
        undated = sum(1 for s in studies if s.start_year is None)
        if undated:
            notes.append(
                f"{undated} study/studies had no usable start date and are "
                "excluded from the trend."
            )

    if top_n is not None and len(buckets) > top_n:
        notes.append(f"Showing the top {top_n} of {len(buckets)} values by study count.")
        buckets = buckets[:top_n]

    return AggregationResult(buckets=buckets, notes=notes)


# Registry: operation -> aggregator. Adding an analysis type is a new function
# plus one entry here; nothing else in the pipeline changes.
AGGREGATORS = {
    AnalysisOp.DISTRIBUTION: distribution,
}


def aggregate(
    operation: AnalysisOp, studies: list[Study], dimension: Dimension, **kwargs
) -> AggregationResult:
    aggregator = AGGREGATORS.get(operation)
    if aggregator is None:
        raise NotImplementedError(f"No aggregator registered for {operation.value}")
    return aggregator(studies, dimension, **kwargs)
