"""Attach source records to aggregated data points.

Pure formatting. The NCT IDs were captured during aggregation, so this module
only selects, bounds, and renders them -- it never decides which studies belong
to a bucket, and the LLM never touches this path at all.
"""

from app.aggregate import Bucket
from app.schemas.api import Citation
from app.schemas.plan import Dimension
from app.schemas.study import Study


def _excerpt(study: Study, dimension: Dimension) -> str:
    """The study field value that justified this study's bucket assignment.

    Quoting the deciding field (rather than a generic snippet) is what makes a
    citation checkable: a reader can open the record and see the same value.
    """
    match dimension:
        case Dimension.PHASE:
            phases = ", ".join(study.phases) if study.phases else "none recorded"
            return f"Phase: {phases}"
        case Dimension.STATUS:
            return f"Overall status: {study.overall_status or 'unknown'}"
        case Dimension.YEAR:
            return f"Start date: {study.start_date or 'not recorded'}"
        case Dimension.COUNTRY:
            return f"Locations: {', '.join(study.countries) or 'none recorded'}"
        case Dimension.SPONSOR:
            return f"Lead sponsor: {study.lead_sponsor or 'not recorded'}"
        case Dimension.SPONSOR_TYPE:
            return f"Sponsor class: {study.sponsor_class or 'not recorded'}"
    return study.brief_title


def build_citations(
    bucket: Bucket,
    studies_by_id: dict[str, Study],
    dimension: Dimension,
    limit: int,
) -> list[Citation]:
    """Render up to `limit` citations for one bucket.

    Bounded deliberately: a bucket may hold hundreds of studies, and returning
    all of them would bloat the payload without adding evidential value.
    """
    citations: list[Citation] = []
    for nct_id in bucket.nct_ids[:limit]:
        study = studies_by_id.get(nct_id)
        if study is None:
            continue
        citations.append(
            Citation(
                nct_id=study.nct_id,
                url=study.url,
                title=study.brief_title,
                excerpt=_excerpt(study, dimension),
            )
        )
    return citations
