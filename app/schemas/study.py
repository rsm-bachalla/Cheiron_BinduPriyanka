"""The normalized Study: a flat view of the deeply-nested API payload.

Everything downstream of normalization reads this and never the raw JSON, so
the API's nesting is confined to a single module.
"""

from pydantic import BaseModel, Field

CTGOV_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


class Intervention(BaseModel):
    """One study intervention.

    `type` is kept because the network analysis needs to separate actual drugs
    from procedures, devices, and behavioural arms -- a distinction that is lost
    if only the name is carried forward.
    """

    name: str
    type: str | None = None  # raw API enum, e.g. "DRUG", "BIOLOGICAL", "DEVICE"


class Study(BaseModel):
    nct_id: str
    brief_title: str = ""
    overall_status: str | None = None
    # Raw API phase enums, e.g. ["PHASE1", "PHASE2"]. Empty for observational
    # studies, which omit the field entirely.
    phases: list[str] = Field(default_factory=list)
    start_date: str | None = None  # raw: "YYYY", "YYYY-MM", or "YYYY-MM-DD"
    start_year: int | None = None  # parsed; None when absent or unparseable
    lead_sponsor: str | None = None
    sponsor_class: str | None = None
    countries: list[str] = Field(default_factory=list)  # deduped, order-preserved
    interventions: list[Intervention] = Field(default_factory=list)
    study_type: str | None = None
    # Set when a study is fetched as part of a comparison fan-out, tagging which
    # group's query produced it.
    group: str | None = None

    @property
    def url(self) -> str:
        return CTGOV_STUDY_URL.format(nct_id=self.nct_id)
