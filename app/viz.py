"""Chart selection and spec construction. Fully deterministic.

Chart type is a function of the analysis operation and the shape of the result,
not an LLM judgement call -- the appropriate chart for "count by category" is
knowable without a model.
"""

from app.aggregate import Bucket
from app.schemas.api import (
    AxisEncoding,
    ChartType,
    Citation,
    Encoding,
    Visualization,
)
from app.schemas.plan import AnalysisOp, Dimension

# The field name each dimension takes in the emitted data rows. The frontend
# reads these via `encoding`, so they stay stable and human-legible.
DIMENSION_FIELD = {
    Dimension.PHASE: "phase",
    Dimension.STATUS: "status",
    Dimension.YEAR: "year",
    Dimension.COUNTRY: "country",
    Dimension.SPONSOR: "sponsor",
    Dimension.SPONSOR_TYPE: "sponsor_type",
}

DIMENSION_TITLE = {
    Dimension.PHASE: "Trial Phase",
    Dimension.STATUS: "Recruitment Status",
    Dimension.YEAR: "Start Year",
    Dimension.COUNTRY: "Country",
    Dimension.SPONSOR: "Lead Sponsor",
    Dimension.SPONSOR_TYPE: "Sponsor Type",
}

VALUE_FIELD = "trial_count"


def select_chart_type(operation: AnalysisOp, dimension: Dimension) -> ChartType:
    """Pick the chart that suits the operation and dimension."""
    if operation is AnalysisOp.TIME_TREND:
        return "line_chart"
    if operation is AnalysisOp.COMPARISON:
        return "grouped_bar_chart"
    if operation is AnalysisOp.NETWORK:
        return "network_graph"
    if operation is AnalysisOp.GEO or dimension is Dimension.COUNTRY:
        # A ranked list of countries; the frontend may render it as a map or a
        # bar chart, and the row shape supports both.
        return "geo_ranking"
    return "bar_chart"


def build_visualization(
    *,
    operation: AnalysisOp,
    dimension: Dimension,
    title: str,
    buckets: list[Bucket],
    citations_by_label: dict[str, list[Citation]],
) -> Visualization:
    """Assemble the frontend-facing spec from aggregated buckets."""
    x_field = DIMENSION_FIELD[dimension]
    x_title = DIMENSION_TITLE[dimension]
    chart_type = select_chart_type(operation, dimension)

    x_type = "temporal" if dimension is Dimension.YEAR else "nominal"

    rows: list[dict] = []
    for bucket in buckets:
        row: dict = {
            x_field: bucket.label,
            VALUE_FIELD: bucket.value,
            "citations": [
                c.model_dump() for c in citations_by_label.get(bucket.label, [])
            ],
        }
        if bucket.group is not None:
            row["series"] = bucket.group
        rows.append(row)

    encoding = Encoding(
        x=AxisEncoding(field=x_field, type=x_type, title=x_title),
        y=AxisEncoding(field=VALUE_FIELD, type="quantitative", title="Number of Trials"),
        series=(
            AxisEncoding(field="series", type="nominal", title="Group")
            if any(b.group is not None for b in buckets)
            else None
        ),
    )

    return Visualization(
        type=chart_type, title=title, encoding=encoding, data=rows
    )
