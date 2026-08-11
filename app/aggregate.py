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


def _dimension_notes(
    studies: list[Study], labels: list[str], dimension: Dimension
) -> list[str]:
    """Caveats implied by how this dimension buckets studies.

    Shared by every count-based aggregator so a comparison discloses exactly the
    same arithmetic as the equivalent single-series distribution.
    """
    notes: list[str] = []

    if dimension is Dimension.PHASE:
        if NO_PHASE_LABEL in labels:
            notes.append(
                f"'{NO_PHASE_LABEL}' covers studies with no phase recorded "
                "(typically observational); it is distinct from 'Not Applicable'."
            )
        if any("/" in label for label in labels):
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

    return notes


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

    notes = _dimension_notes(studies, [b.label for b in buckets], dimension)

    if top_n is not None and len(buckets) > top_n:
        notes.append(f"Showing the top {top_n} of {len(buckets)} values by study count.")
        buckets = buckets[:top_n]

    return AggregationResult(buckets=buckets, notes=notes)


def comparison(
    studies: list[Study],
    dimension: Dimension,
    groups: list[str] | None = None,
    top_n: int | None = None,
) -> AggregationResult:
    """Count studies per `dimension` value, held separately per comparison group.

    Studies arrive already tagged with the group whose upstream query returned
    them (see the fan-out in `pipeline`), so this is the same counting as
    `distribution` partitioned by that tag -- no membership is inferred here.

    The output is dense: every (label, group) pair is emitted, with zeros where a
    group has no studies for a label. A grouped bar chart needs the absence to be
    a visible zero rather than a missing row.
    """
    groups = groups or sorted({s.group for s in studies if s.group})

    per_group: dict[str, dict[str, list[str]]] = {g: defaultdict(list) for g in groups}
    for study in studies:
        counts = per_group.get(study.group or "")
        if counts is None:
            continue
        for label in _labels_for(study, dimension):
            counts[label].append(study.nct_id)

    # Label order comes from the combined totals, so both series share one axis
    # ordered the same way a single-series chart would be.
    totals: dict[str, int] = defaultdict(int)
    for counts in per_group.values():
        for label, ids in counts.items():
            totals[label] += len(ids)

    ordered = _sort_buckets(
        [Bucket(label=label, value=value) for label, value in totals.items()], dimension
    )
    labels = [bucket.label for bucket in ordered]

    notes = _dimension_notes(studies, labels, dimension)
    notes.append(
        "Each group was queried against ClinicalTrials.gov independently; a "
        "trial matching more than one group is counted once in each."
    )

    # The cap applies to categories, not series, so groups stay comparable.
    if top_n is not None and len(labels) > top_n:
        notes.append(
            f"Showing the top {top_n} of {len(labels)} values by combined study count."
        )
        labels = labels[:top_n]

    empty = [g for g in groups if not per_group[g]]
    if empty:
        notes.append(
            f"No trials were returned for: {', '.join(empty)}. "
            "These groups are shown as zero rather than omitted."
        )

    buckets = [
        Bucket(
            label=label,
            value=len(per_group[group].get(label, [])),
            nct_ids=per_group[group].get(label, []),
            group=group,
        )
        for label in labels
        for group in groups
    ]

    return AggregationResult(buckets=buckets, notes=notes)


# Registry: operation -> aggregator. Adding an analysis type is a new function
# plus one entry here; nothing else in the pipeline changes.
#
# TIME_TREND and GEO are counts over a different axis rather than different
# arithmetic, so they reuse `distribution`. They stay separate operations
# because they select different charts and carry different caveats.
#
# NETWORK is deliberately absent: it produces nodes and edges rather than
# buckets, so it lives in `network.py` with its own result type instead of being
# forced through this signature.
AGGREGATORS = {
    AnalysisOp.DISTRIBUTION: distribution,
    AnalysisOp.TIME_TREND: distribution,
    AnalysisOp.GEO: distribution,
    AnalysisOp.COMPARISON: comparison,
}


def aggregate(
    operation: AnalysisOp, studies: list[Study], dimension: Dimension, **kwargs
) -> AggregationResult:
    aggregator = AGGREGATORS.get(operation)
    if aggregator is None:
        raise NotImplementedError(f"No aggregator registered for {operation.value}")
    return aggregator(studies, dimension, **kwargs)
