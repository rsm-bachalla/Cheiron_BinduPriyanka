"""Streamlit demo: a thin HTTP client for the Clinical Trials Insight API.

This exists to prove one claim -- that the API's response is genuinely
frontend-renderable -- so it deliberately contains no analytics. It never
imports the planner, the aggregators, the ClinicalTrials.gov client, or any
backend module; FastAPI is an external service reached over HTTP.

The stronger version of that claim is that this file never hardcodes a field
name from a chart row. Every renderer reads `visualization.encoding` to learn
which key holds the category, the value, and the series, so a new analysis type
that emits the same envelope renders here without a code change.
"""

import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

# Citations are capped for display only. The backend response is kept intact in
# session state, and the raw JSON is one expander away.
MAX_CITATIONS_SHOWN = 25
MAX_GEO_BARS = 20

EXAMPLES = {
    "Distribution": "How are breast cancer trials distributed across phases?",
    "Time trend": "How has the number of trials for pembrolizumab changed over time?",
    "Geographic": "Which countries have the most recruiting trials for lung cancer?",
    "Comparison": "Compare trial phases for pembrolizumab vs nivolumab",
    "Network": "Show a network of sponsors and drugs for melanoma trials",
}


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------


def call_api(query: str, api_base_url: str) -> tuple[dict | None, str | None]:
    """POST /query. Returns (body, error_message) -- exactly one is populated.

    Every failure mode the backend can produce is translated here into a
    message a human can act on, while keeping the machine-readable detail
    visible for debugging.
    """
    try:
        response = httpx.post(
            f"{api_base_url.rstrip('/')}/query",
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.ConnectError:
        return None, (
            f"**Cannot reach the backend at `{api_base_url}`.**\n\n"
            "Start it in another terminal with `make run`, or point this demo "
            "elsewhere with `API_BASE_URL=... streamlit run demo/streamlit_app.py`."
        )
    except httpx.TimeoutException:
        return None, (
            f"**The backend did not respond within {REQUEST_TIMEOUT:.0f}s.** "
            "Broad queries fetch up to 1000 records from ClinicalTrials.gov; "
            "try a narrower question."
        )
    except httpx.HTTPError as exc:
        return None, f"**Request failed:** `{type(exc).__name__}: {exc}`"

    if response.status_code == 200:
        try:
            body = response.json()
        except ValueError:
            return None, "**The backend returned a 200 that was not JSON.**"
        if "visualization" not in body or "meta" not in body:
            return None, (
                "**Unexpected response shape:** a 200 without `visualization` "
                f"and `meta`. Keys received: `{sorted(body)}`."
            )
        return body, None

    return None, _describe_error(response)


def _describe_error(response: httpx.Response) -> str:
    """Turn a non-200 into a friendly message that still names the cause."""
    try:
        payload = response.json()
    except ValueError:
        return (
            f"**HTTP {response.status_code}** from the backend.\n\n"
            f"```\n{response.text[:500]}\n```"
        )

    # FastAPI's own request-validation failures use `detail`, not our contract.
    if "error" not in payload:
        return (
            f"**HTTP {response.status_code}** — the request itself was "
            f"rejected.\n\n```json\n{payload}\n```"
        )

    error = payload["error"]
    headline = {
        422: "That question could not be mapped to a supported analysis.",
        404: "The question was understood, but no trials matched.",
        501: "The question was understood, but that analysis is not built yet.",
        502: "ClinicalTrials.gov rejected the request or was unreachable.",
    }.get(response.status_code, f"The backend returned HTTP {response.status_code}.")

    lines = [
        f"**{headline}**",
        "",
        f"{error.get('message', '(no message)')}",
        "",
        f"`{error.get('code', 'unknown')}` · HTTP {response.status_code}",
    ]

    details = error.get("details") or {}
    for key in ("reason", "interpreted_intent", "failed_group", "supported"):
        if value := details.get(key):
            lines.append(f"\n**{key.replace('_', ' ').title()}:** {value}")
    if remaining := {k: v for k, v in details.items() if k not in
                     {"reason", "interpreted_intent", "failed_group", "supported"}}:
        lines.append(f"\n**Other details:** `{remaining}`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderers -- all driven by `encoding`, none by hardcoded field names
# ---------------------------------------------------------------------------


def _fields(viz: dict) -> tuple[str | None, str | None, str | None]:
    """(category, value, series) field names, read from the declared encoding."""
    encoding = viz.get("encoding") or {}
    x = (encoding.get("x") or {}).get("field")
    y = (encoding.get("y") or {}).get("field")
    series = (encoding.get("series") or {}).get("field")
    return x, y, series


def _axis_title(viz: dict, axis: str, fallback: str) -> str:
    return ((viz.get("encoding") or {}).get(axis) or {}).get("title") or fallback


def _rows_frame(viz: dict) -> pd.DataFrame:
    """Chart rows minus the citation payload, which is rendered separately."""
    return pd.DataFrame(
        [{k: v for k, v in row.items() if k != "citations"} for row in viz["data"]]
    )


def render_visualization(viz: dict) -> None:
    """Dispatch on the declared chart type."""
    chart_type = viz.get("type")

    if chart_type == "network_graph":
        render_network(viz)
        return

    if not viz.get("data"):
        st.info("The backend returned no data points for this query.")
        return

    x_field, y_field, series_field = _fields(viz)
    if not x_field or not y_field:
        st.warning(
            "The response did not declare x/y encodings, so it cannot be "
            "rendered generically. Raw rows below."
        )
        st.dataframe(_rows_frame(viz), width="stretch")
        return

    frame = _rows_frame(viz)

    if chart_type == "grouped_bar_chart" and series_field:
        render_grouped_bar(viz, frame, x_field, y_field, series_field)
    elif chart_type == "geo_ranking":
        render_geo_ranking(viz, frame, x_field, y_field)
    elif chart_type == "line_chart":
        # Streamlit-native: the row order from the backend is already the axis
        # order, and a year index is exactly what st.line_chart wants.
        st.line_chart(
            frame.set_index(x_field)[y_field],
            x_label=_axis_title(viz, "x", x_field),
            y_label=_axis_title(viz, "y", y_field),
        )
    else:
        st.bar_chart(
            frame.set_index(x_field)[y_field],
            x_label=_axis_title(viz, "x", x_field),
            y_label=_axis_title(viz, "y", y_field),
        )

    with st.expander(f"Data table ({len(frame)} rows)"):
        st.dataframe(frame, width="stretch", hide_index=True)


def render_grouped_bar(
    viz: dict, frame: pd.DataFrame, x_field: str, y_field: str, series_field: str
) -> None:
    """Side-by-side bars per series. Altair, because `xOffset` has no native equivalent."""
    order = list(dict.fromkeys(frame[x_field]))  # backend order is meaningful
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_field}:N", sort=order, title=_axis_title(viz, "x", x_field)),
            xOffset=f"{series_field}:N",
            y=alt.Y(f"{y_field}:Q", title=_axis_title(viz, "y", y_field)),
            color=alt.Color(
                f"{series_field}:N", title=_axis_title(viz, "series", "Group")
            ),
            tooltip=list(frame.columns),
        )
        .properties(height=420)
    )
    st.altair_chart(chart, width="stretch")


def render_geo_ranking(
    viz: dict, frame: pd.DataFrame, x_field: str, y_field: str
) -> None:
    """Horizontal ranked bars. Altair, to keep the backend's descending order."""
    shown = frame.head(MAX_GEO_BARS)
    chart = (
        alt.Chart(shown)
        .mark_bar()
        .encode(
            y=alt.Y(
                f"{x_field}:N",
                sort=list(shown[x_field]),
                title=_axis_title(viz, "x", x_field),
            ),
            x=alt.X(f"{y_field}:Q", title=_axis_title(viz, "y", y_field)),
            tooltip=list(shown.columns),
        )
        .properties(height=max(320, 26 * len(shown)))
    )
    st.altair_chart(chart, width="stretch")
    if len(frame) > MAX_GEO_BARS:
        st.caption(f"Showing the top {MAX_GEO_BARS} of {len(frame)} values.")


def render_network(viz: dict) -> None:
    """Sponsor-drug graph as a bipartite diagram, plus the edge table.

    Bipartite rather than force-directed on purpose: the graph has exactly two
    node kinds and every edge crosses between them, so sponsors-left /
    drugs-right is both readable and deterministic, where a force layout would
    be a hairball that moves on every render. Drawn in Altair, which Streamlit
    already depends on -- no graph library is pulled in for one view.
    """
    nodes = viz.get("nodes") or []
    edges = viz.get("edges") or []

    if not nodes or not edges:
        st.info("The backend returned no network nodes or edges for this query.")
        return

    sponsors = [n for n in nodes if n.get("node_type") == "sponsor"]
    drugs = [n for n in nodes if n.get("node_type") == "drug"]
    if not sponsors or not drugs:
        st.warning("The network is missing one side of the graph; showing edges only.")
    else:
        st.altair_chart(_bipartite_chart(sponsors, drugs, edges), width="stretch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sponsors", len(sponsors))
    col_b.metric("Drugs", len(drugs))
    col_c.metric("Relationships", len(edges))

    edge_frame = pd.DataFrame(
        [
            {
                "sponsor": e.get("source"),
                "drug": e.get("target"),
                "trial_count": e.get("trial_count"),
                "sources": len(e.get("citations") or []),
            }
            for e in edges
        ]
    )
    st.markdown("**Sponsor-drug relationships**")
    st.dataframe(edge_frame, width="stretch", hide_index=True)


def _bipartite_chart(sponsors: list[dict], drugs: list[dict], edges: list[dict]):
    """Sponsors on the left, drugs on the right, edges weighted by trial count."""

    def positions(items: list[dict], x: float) -> dict[str, dict]:
        # Heaviest nodes toward the top; a single node sits centred.
        ranked = sorted(items, key=lambda n: -(n.get("trial_count") or 0))
        span = max(len(ranked) - 1, 1)
        return {
            node["id"]: {
                "x": x,
                "y": 1 - (i / span) if len(ranked) > 1 else 0.5,
                "label": node.get("label", node["id"]),
                "trial_count": node.get("trial_count"),
                "node_type": node.get("node_type"),
            }
            for i, node in enumerate(ranked)
        }

    coords = positions(sponsors, 0.0) | positions(drugs, 1.0)

    edge_rows = [
        {
            "x": coords[e["source"]]["x"],
            "y": coords[e["source"]]["y"],
            "x2": coords[e["target"]]["x"],
            "y2": coords[e["target"]]["y"],
            "sponsor": e["source"],
            "drug": e["target"],
            "trial_count": e.get("trial_count"),
        }
        for e in edges
        if e.get("source") in coords and e.get("target") in coords
    ]
    node_rows = list(coords.values())

    height = max(420, 22 * max(len(sponsors), len(drugs)))
    hidden_axis = alt.Axis(labels=False, ticks=False, title=None, domain=False)
    # Padded so the text labels have room outside the two node columns.
    x_scale = alt.Scale(domain=[-0.75, 1.75])
    y_scale = alt.Scale(domain=[-0.05, 1.05])

    edge_layer = (
        alt.Chart(pd.DataFrame(edge_rows))
        .mark_rule(opacity=0.35, color="#7f9fbf")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=hidden_axis),
            y=alt.Y("y:Q", scale=y_scale, axis=hidden_axis),
            x2="x2:Q",
            y2="y2:Q",
            strokeWidth=alt.StrokeWidth(
                "trial_count:Q", scale=alt.Scale(range=[0.5, 6]), legend=None
            ),
            tooltip=["sponsor:N", "drug:N", "trial_count:Q"],
        )
    )

    node_frame = pd.DataFrame(node_rows)
    point_layer = (
        alt.Chart(node_frame)
        .mark_circle()
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=hidden_axis),
            y=alt.Y("y:Q", scale=y_scale, axis=hidden_axis),
            size=alt.Size("trial_count:Q", scale=alt.Scale(range=[40, 400]), legend=None),
            color=alt.Color("node_type:N", title="Node type"),
            tooltip=["label:N", "node_type:N", "trial_count:Q"],
        )
    )

    def labels(node_type: str, align: str, dx: int):
        return (
            alt.Chart(node_frame[node_frame["node_type"] == node_type])
            .mark_text(align=align, dx=dx, fontSize=10)
            .encode(
                x=alt.X("x:Q", scale=x_scale, axis=hidden_axis),
                y=alt.Y("y:Q", scale=y_scale, axis=hidden_axis),
                text="label:N",
            )
        )

    return (
        edge_layer
        + point_layer
        + labels("sponsor", "right", -8)
        + labels("drug", "left", 8)
    ).properties(height=height).configure_view(stroke=None)


# ---------------------------------------------------------------------------
# Metadata and citations
# ---------------------------------------------------------------------------


def render_meta(meta: dict) -> None:
    interpretation = meta.get("query_interpretation") or {}

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Planner", interpretation.get("planner", "—"))
    col_b.metric("Analysis", interpretation.get("operation", "—"))
    col_c.metric("Grouped by", interpretation.get("dimension", "—"))
    col_d.metric("Trials analysed", f"{meta.get('record_count', 0):,}")

    if intent := interpretation.get("intent"):
        st.markdown(f"**Interpreted as:** {intent}")

    if filters := meta.get("filters"):
        st.markdown(
            "**Filters:** "
            + " · ".join(f"`{key}` = {value}" for key, value in filters.items())
        )
    else:
        st.markdown("**Filters:** _none_")

    total = meta.get("total_available")
    if meta.get("truncated"):
        # total_available is absent for comparisons, where a combined total has
        # no defensible meaning -- per-group totals are shown below instead.
        scope = f"of {total:,} matching" if total is not None else "of the full match set"
        st.warning(
            f"**Truncated:** analysed {meta.get('record_count', 0):,} {scope} "
            "trials. Counts are a sample, not registry totals."
        )
    elif total is not None:
        st.success(f"**Complete:** all {total:,} matching trials were analysed.")

    if groups := meta.get("groups"):
        st.markdown("**Per comparison group**")
        st.dataframe(
            pd.DataFrame(groups).rename(columns={"group": "comparison group"}),
            width="stretch",
            hide_index=True,
        )

    if notes := meta.get("notes"):
        st.markdown("**How these numbers were produced**")
        for note in notes:
            st.markdown(f"- {note}")


def collect_citations(viz: dict) -> list[dict]:
    """Every citation in the response, deduplicated by NCT ID, order preserved."""
    carriers = list(viz.get("data") or []) + list(viz.get("edges") or [])
    seen: set[str] = set()
    citations: list[dict] = []
    for carrier in carriers:
        for citation in carrier.get("citations") or []:
            nct_id = citation.get("nct_id")
            if not nct_id or nct_id in seen:
                continue
            seen.add(nct_id)
            citations.append(citation)
    return citations


def render_citations(viz: dict) -> None:
    citations = collect_citations(viz)
    if not citations:
        return

    label = f"Sources / ClinicalTrials.gov records ({len(citations)} distinct)"
    with st.expander(label):
        st.caption(
            "Every data point above carries the records that produced it. The "
            "excerpt quotes the field value that placed the trial in that bucket."
        )
        for citation in citations[:MAX_CITATIONS_SHOWN]:
            st.markdown(
                f"**[{citation.get('nct_id', '?')}]"
                f"({citation.get('url', '#')})** — {citation.get('title', '')}"
            )
            st.caption(citation.get("excerpt", ""))
        if len(citations) > MAX_CITATIONS_SHOWN:
            st.caption(
                f"Showing {MAX_CITATIONS_SHOWN} of {len(citations)} distinct "
                "records. The full set is in the raw response below."
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Clinical Trials Insight Agent", layout="wide")

st.title("Clinical Trials Insight Agent")
st.caption(
    "Ask a question in plain English. The backend interprets the intent, queries "
    "ClinicalTrials.gov, and returns a chart specification with every data point "
    "traceable to the trial records behind it."
)

with st.sidebar:
    st.subheader("Backend")
    api_base_url = st.text_input("API base URL", value=API_BASE_URL)
    st.caption("Override with the `API_BASE_URL` environment variable.")
    try:
        health = httpx.get(f"{api_base_url.rstrip('/')}/health", timeout=3.0)
        if health.status_code == 200:
            st.success("Connected")
        else:
            st.warning(f"Responded HTTP {health.status_code}")
    except httpx.HTTPError:
        st.error("Unreachable — run `make run`")

    st.divider()
    st.caption(
        "This demo is a pure HTTP client. It imports no backend module and "
        "performs no filtering, counting, or chart selection of its own."
    )

st.session_state.setdefault("query", EXAMPLES["Distribution"])

st.markdown("**Try an example**")
for column, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
    if column.button(label, width="stretch"):
        st.session_state["query"] = example

query = st.text_area("Your question", key="query", height=90)
submitted = st.button("Run query", type="primary")

if submitted:
    if not query or len(query.strip()) < 3:
        st.error("Please enter a question of at least 3 characters.")
    else:
        with st.spinner("Interpreting the question and querying ClinicalTrials.gov…"):
            body, error = call_api(query.strip(), api_base_url)
        st.session_state["result"] = body
        st.session_state["error"] = error

if st.session_state.get("error"):
    st.error(st.session_state["error"])

if result := st.session_state.get("result"):
    visualization = result["visualization"]

    st.divider()
    st.subheader(visualization.get("title", "Result"))
    st.caption(f"Chart type: `{visualization.get('type')}`")

    render_visualization(visualization)

    st.divider()
    render_meta(result["meta"])
    render_citations(visualization)

    with st.expander("Raw API response"):
        st.json(result)
