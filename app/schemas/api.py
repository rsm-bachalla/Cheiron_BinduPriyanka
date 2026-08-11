"""Public request/response contract.

This is what a frontend codes against, so it stays stable and self-describing:
chart type, encoding, and rows that already carry their own citations.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.plan import TrialPhase, TrialStatus

ChartType = Literal[
    "bar_chart",
    "line_chart",
    "grouped_bar_chart",
    "geo_ranking",
    "network_graph",
]


class QueryHints(BaseModel):
    """Optional structured overrides.

    When a hint is set it *wins* over whatever intent interpretation produced,
    so a caller can always force a deterministic filter.
    """

    drug_name: str | None = None
    condition: str | None = None
    trial_phase: TrialPhase | None = None
    sponsor: str | None = None
    country: str | None = None
    status: TrialStatus | None = None
    start_year: int | None = Field(None, ge=1900, le=2100)
    end_year: int | None = Field(None, ge=1900, le=2100)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    hints: QueryHints | None = None


class Citation(BaseModel):
    """A pointer back to the exact ClinicalTrials.gov record behind a data point."""

    nct_id: str
    url: str
    title: str
    excerpt: str = Field(
        ..., description="The study field value that placed it in this bucket"
    )


class AxisEncoding(BaseModel):
    field: str
    type: Literal["nominal", "quantitative", "temporal", "ordinal"]
    title: str


class Encoding(BaseModel):
    x: AxisEncoding | None = None
    y: AxisEncoding | None = None
    series: AxisEncoding | None = None


class Visualization(BaseModel):
    type: ChartType
    title: str
    encoding: Encoding
    # Chart-ready rows. Field names match the encoding above. Each row carries a
    # `citations` list, so traceability survives all the way to a rendered point.
    data: list[dict[str, Any]] = Field(default_factory=list)
    # Populated only for network_graph, which needs nodes+edges rather than rows.
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None


class QueryInterpretation(BaseModel):
    intent: str
    operation: str
    dimension: str
    planner: str = Field(..., description="Which planner produced this: rulebased | openai")


class GroupMeta(BaseModel):
    """Per-group provenance for a comparison.

    Each group is a separate upstream query, so each has its own match count and
    its own truncation state. Reporting them separately is what keeps a
    comparison honest: one group capped at 1000 and another returning 40 is not
    a like-for-like chart, and the caller can see that.
    """

    group: str
    record_count: int
    total_available: int | None = None
    truncated: bool = False


class Meta(BaseModel):
    query_interpretation: QueryInterpretation
    filters: dict[str, Any]
    source: str = "ClinicalTrials.gov"
    source_api: str = "https://clinicaltrials.gov/api/v2/studies"
    # Studies actually analysed.
    record_count: int
    # Studies matching upstream. Exceeds record_count when the fetch cap
    # truncates. Deliberately null for comparisons -- groups are fetched by
    # separate queries whose match sets can overlap, so a combined total would
    # be a number with no defensible meaning. See `groups` instead.
    total_available: int | None = None
    truncated: bool = False
    # Populated only for comparisons: one entry per group, in chart series order.
    groups: list[GroupMeta] | None = None
    # Analytical caveats: multi-phase handling, multi-country double counting,
    # undated studies excluded, caps applied. Always populated where relevant.
    notes: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    visualization: Visualization
    meta: Meta
