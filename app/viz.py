"""Chart selection and spec construction. Fully deterministic.

Chart type is a function of the analysis operation and the shape of the result,
not an LLM judgement call -- the appropriate chart for "count by category" is
knowable without a model.
"""

from dataclasses import asdict

from app.aggregate import Bucket
from app.network import Edge, Node
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
# The row key identifying which comparison series a point belongs to.
SERIES_FIELD = "group"


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
    citations: list[list[Citation]],
) -> Visualization:
    """Assemble the frontend-facing spec from aggregated buckets.

    `citations` is positional -- one list per bucket, in bucket order. Keying by
    label would collide across comparison series, where the same label appears
    once per group.
    """
    x_field = DIMENSION_FIELD[dimension]
    x_title = DIMENSION_TITLE[dimension]
    chart_type = select_chart_type(operation, dimension)

    x_type = "temporal" if dimension is Dimension.YEAR else "nominal"

    rows: list[dict] = []
    for bucket, bucket_citations in zip(buckets, citations, strict=True):
        row: dict = {
            x_field: bucket.label,
            VALUE_FIELD: bucket.value,
            "citations": [c.model_dump() for c in bucket_citations],
        }
        if bucket.group is not None:
            row[SERIES_FIELD] = bucket.group
        rows.append(row)

    encoding = Encoding(
        x=AxisEncoding(field=x_field, type=x_type, title=x_title),
        y=AxisEncoding(field=VALUE_FIELD, type="quantitative", title="Number of Trials"),
        series=(
            AxisEncoding(field=SERIES_FIELD, type="nominal", title="Group")
            if any(b.group is not None for b in buckets)
            else None
        ),
    )

    return Visualization(
        type=chart_type, title=title, encoding=encoding, data=rows
    )


def build_network_visualization(
    *,
    title: str,
    nodes: list[Node],
    edges: list[Edge],
    citations: list[list[Citation]],
) -> Visualization:
    """Assemble a node/edge spec.

    A relationship graph has no x/y rows, so forcing it into `data` would leave
    the frontend reconstructing the topology. `nodes` and `edges` are explicit
    instead, with citations on the edges -- the edge is the claim being made.
    """
    return Visualization(
        type="network_graph",
        title=title,
        # `trial_count` means the same thing on a node and on an edge, so one
        # quantitative encoding describes both.
        encoding=Encoding(
            y=AxisEncoding(
                field=VALUE_FIELD, type="quantitative", title="Number of Trials"
            )
        ),
        data=[],
        nodes=[asdict(node) for node in nodes],
        edges=[
            {
                "source": edge.source,
                "target": edge.target,
                VALUE_FIELD: edge.trial_count,
                "citations": [c.model_dump() for c in edge_citations],
            }
            for edge, edge_citations in zip(edges, citations, strict=True)
        ],
    )
