"""Raw ClinicalTrials.gov JSON -> flat, typed `Study` objects.

All knowledge of the API's `protocolSection` nesting lives here. If the upstream
shape changes, this is the only module that moves.
"""

import logging

from app.schemas.study import Study

logger = logging.getLogger(__name__)

# Display labels for the raw phase enums. Verified against live responses; the
# API also omits `phases` entirely for observational studies, handled below.
PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}

# Studies that omit `phases` are observational; they are not "Not Applicable"
# (which is a real interventional value) and must stay distinguishable.
NO_PHASE_LABEL = "Not Specified"

STATUS_LABELS = {
    "NOT_YET_RECRUITING": "Not Yet Recruiting",
    "RECRUITING": "Recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by Invitation",
    "ACTIVE_NOT_RECRUITING": "Active, Not Recruiting",
    "SUSPENDED": "Suspended",
    "TERMINATED": "Terminated",
    "COMPLETED": "Completed",
    "WITHDRAWN": "Withdrawn",
    "UNKNOWN": "Unknown",
}


def parse_start_year(raw_date: str | None) -> int | None:
    """Extract a year from a ClinicalTrials.gov date.

    The API emits three widths -- "2022", "2022-08", "2022-08-11" -- and the year
    is the leading token in all of them, which is all any time bucketing needs.
    """
    if not raw_date or len(raw_date) < 4:
        return None
    head = raw_date[:4]
    if not head.isdigit():
        return None
    year = int(head)
    # Guard against malformed values reaching the aggregation layer.
    return year if 1900 <= year <= 2100 else None


def phase_label(phases: list[str]) -> str:
    """Collapse a study's phase list into one display label.

    A study registered as PHASE1+PHASE2 becomes the distinct bucket
    "Phase 1/Phase 2" rather than incrementing both, so bucket counts always sum
    to the study count.
    """
    if not phases:
        return NO_PHASE_LABEL
    return "/".join(PHASE_LABELS.get(p, p.title()) for p in phases)


def status_label(status: str | None) -> str:
    if not status:
        return "Unknown"
    return STATUS_LABELS.get(status, status.replace("_", " ").title())


def normalize_study(raw: dict, group: str | None = None) -> Study | None:
    """Flatten one raw study. Returns None if it has no NCT ID to cite."""
    protocol = raw.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}

    nct_id = identification.get("nctId")
    if not nct_id:
        # Without an ID the record cannot be cited, so it cannot be used.
        logger.warning("Skipping study with no nctId")
        return None

    status_module = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    sponsor_module = protocol.get("sponsorCollaboratorsModule") or {}
    lead_sponsor = sponsor_module.get("leadSponsor") or {}
    locations = (protocol.get("contactsLocationsModule") or {}).get("locations") or []
    interventions = (protocol.get("armsInterventionsModule") or {}).get(
        "interventions"
    ) or []

    # A trial runs at many sites in the same country; dedupe while preserving
    # order so a trial contributes at most once per country.
    countries: list[str] = []
    for location in locations:
        country = location.get("country")
        if country and country not in countries:
            countries.append(country)

    start_date = (status_module.get("startDateStruct") or {}).get("date")

    return Study(
        nct_id=nct_id,
        brief_title=identification.get("briefTitle") or "",
        overall_status=status_module.get("overallStatus"),
        phases=design.get("phases") or [],
        start_date=start_date,
        start_year=parse_start_year(start_date),
        lead_sponsor=lead_sponsor.get("name"),
        sponsor_class=lead_sponsor.get("class"),
        countries=countries,
        interventions=[i["name"] for i in interventions if i.get("name")],
        study_type=design.get("studyType"),
        group=group,
    )


def normalize_studies(raw_studies: list[dict], group: str | None = None) -> list[Study]:
    normalized = (normalize_study(raw, group) for raw in raw_studies)
    return [study for study in normalized if study is not None]
