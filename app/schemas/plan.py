"""The QueryPlan: the contract between intent interpretation and everything else.

This is the *only* structure the LLM ever produces. Every field is enum- or
type-constrained, so a hallucinated value fails validation rather than silently
corrupting a ClinicalTrials.gov query.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AnalysisOp(str, Enum):
    """The deterministic analysis to run over the fetched studies."""

    TIME_TREND = "time_trend"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    GEO = "geo"
    NETWORK = "network"


class Dimension(str, Enum):
    """The study attribute to group by."""

    PHASE = "phase"
    STATUS = "status"
    YEAR = "year"
    COUNTRY = "country"
    SPONSOR = "sponsor"
    SPONSOR_TYPE = "sponsor_type"


class TrialPhase(str, Enum):
    """ClinicalTrials.gov phase enum, verified against live API responses."""

    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NA = "NA"


class TrialStatus(str, Enum):
    """ClinicalTrials.gov overallStatus enum.

    The API returns HTTP 400 with a plain-text body for unknown values, so we
    constrain these locally rather than discovering the failure upstream.
    """

    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class TrialFilters(BaseModel):
    """Filters applied at the ClinicalTrials.gov query layer.

    Every field here maps to an Essie `AREA[...]` expression or a native filter
    parameter, so filtering happens upstream rather than by pulling everything
    and discarding locally.
    """

    drug: str | None = Field(None, description="Intervention name, e.g. 'pembrolizumab'")
    condition: str | None = Field(None, description="Condition, e.g. 'breast cancer'")
    sponsor: str | None = Field(None, description="Lead sponsor name")
    country: str | None = Field(None, description="Location country")
    phase: TrialPhase | None = None
    status: TrialStatus | None = None
    start_year: int | None = Field(None, ge=1900, le=2100)
    end_year: int | None = Field(None, ge=1900, le=2100)

    @model_validator(mode="after")
    def _check_year_order(self) -> "TrialFilters":
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not be after end_year")
        return self

    def is_empty(self) -> bool:
        """True when no filter at all is set.

        An unfiltered query would scan the entire registry, so callers treat
        this as a refusal condition rather than running it.
        """
        return not any(self.model_dump(exclude_none=True).values())

    def describe(self) -> dict[str, str | int]:
        """Filters as a flat, JSON-friendly dict for the response meta."""
        out: dict[str, str | int] = {}
        for key, value in self.model_dump(exclude_none=True).items():
            out[key] = value.value if isinstance(value, Enum) else value
        return out


class QueryPlan(BaseModel):
    """A fully-specified, executable analysis. Produced once per request."""

    intent: str = Field(..., description="Plain-language restatement of the question")
    operation: AnalysisOp
    dimension: Dimension
    filters: TrialFilters = Field(default_factory=TrialFilters)
    comparison_groups: list[str] = Field(
        default_factory=list,
        description="Drug/entity names to compare; each is fetched separately",
    )
    title: str = Field(..., description="Human-readable chart title")

    @model_validator(mode="after")
    def _check_comparison_groups(self) -> "QueryPlan":
        if self.operation is AnalysisOp.COMPARISON and len(self.comparison_groups) < 2:
            raise ValueError("comparison requires at least two comparison_groups")
        return self
