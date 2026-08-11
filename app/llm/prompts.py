"""Planner prompt.

Deliberately compact. The JSON schema already constrains shape and enum values,
so the prompt only has to convey what the operations and dimensions *mean* and
where the model's authority ends.
"""

PLANNER_SYSTEM_PROMPT = """\
You translate questions about clinical trials into a structured analysis plan.
You do not answer the question.

Choose one operation:
- distribution: how trials break down across a categorical attribute
- time_trend: how trial counts change across years
- comparison: two or more named drugs/entities measured side by side
- geo: which countries or regions have the most trials
- network: relationships between entities, e.g. sponsors and the drugs they run

Choose one dimension (the attribute to group by):
- phase, status, year, country, sponsor, sponsor_type

Fill filters only with values the question actually states:
- drug: an intervention name (e.g. "pembrolizumab")
- condition: a disease (e.g. "breast cancer")
- sponsor, country: names as written
- phase, status: only the allowed enum values
- start_year / end_year: only if the question gives a period

Rules:
- Return only the plan. Never return trial counts, statistics, study IDs, or findings.
- Never estimate or calculate anything. You do not have the data.
- Use comparison_groups only for `comparison`, listing each entity separately.
- Do not invent filters the question does not mention.
- title is a short chart title.
- intent is one plain sentence restating the question.

Pair operations with a sensible dimension: time_trend with year, geo with
country, network with sponsor. If the question is too vague to map (no clear
subject or no clear grouping), still return your best plan -- it is validated
downstream and rejected if incoherent.\
"""


# Few-shot examples covering every operation plus the multi-filter case.
# Kept short: the schema does the structural work.
PLANNER_EXAMPLES = [
    (
        "How are breast cancer trials distributed across phases?",
        {
            "intent": "Break down breast cancer trials by phase.",
            "operation": "distribution",
            "dimension": "phase",
            "filters": {"condition": "breast cancer"},
            "comparison_groups": [],
            "title": "Breast Cancer Trials by Phase",
        },
    ),
    (
        "How has the number of trials for pembrolizumab changed over time?",
        {
            "intent": "Track pembrolizumab trial counts by start year.",
            "operation": "time_trend",
            "dimension": "year",
            "filters": {"drug": "pembrolizumab"},
            "comparison_groups": [],
            "title": "Pembrolizumab Trials Over Time",
        },
    ),
    (
        "Which countries have the most recruiting trials for lung cancer?",
        {
            "intent": "Rank countries by recruiting lung cancer trials.",
            "operation": "geo",
            "dimension": "country",
            "filters": {"condition": "lung cancer", "status": "RECRUITING"},
            "comparison_groups": [],
            "title": "Recruiting Lung Cancer Trials by Country",
        },
    ),
    (
        "Compare trial phases for pembrolizumab vs nivolumab",
        {
            "intent": "Compare phase distribution of pembrolizumab and nivolumab trials.",
            "operation": "comparison",
            "dimension": "phase",
            "filters": {},
            "comparison_groups": ["pembrolizumab", "nivolumab"],
            "title": "Pembrolizumab vs Nivolumab by Phase",
        },
    ),
    (
        "Show a network of sponsors and drugs for melanoma trials",
        {
            "intent": "Map sponsors to the drugs they study in melanoma trials.",
            "operation": "network",
            "dimension": "sponsor",
            "filters": {"condition": "melanoma"},
            "comparison_groups": [],
            "title": "Melanoma Sponsor-Drug Network",
        },
    ),
    (
        "Phase 3 melanoma trials in France since 2019, broken down by status",
        {
            "intent": "Break down phase 3 melanoma trials in France since 2019 by status.",
            "operation": "distribution",
            "dimension": "status",
            "filters": {
                "condition": "melanoma",
                "country": "France",
                "phase": "PHASE3",
                "start_year": 2019,
            },
            "comparison_groups": [],
            "title": "Phase 3 Melanoma Trials in France by Status",
        },
    ),
]


def build_user_prompt(query: str, repair_hint: str | None = None) -> str:
    """Render the few-shot block plus the question under analysis."""
    import json

    # Examples are pretty-printed and clearly delimited. A compact one-line
    # `A: {...}` form led a model to copy JSON punctuation into a filter value
    # ("pembrolizumab},"), so the separation here is deliberate.
    lines = ["Worked examples:", ""]
    for question, plan in PLANNER_EXAMPLES:
        lines.append(f"Question: {question}")
        lines.append("Plan:")
        lines.append(json.dumps(plan, indent=2))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Now plan this question. Filter values must contain only the entity "
        "name itself, with no punctuation or JSON syntax."
    )
    lines.append(f"Question: {query}")

    if repair_hint:
        # Only present on the single repair attempt, naming the exact failure.
        lines.append("")
        lines.append(
            f"Your previous plan was rejected: {repair_hint}\n"
            "Return a corrected plan that satisfies the schema and the rules."
        )

    return "\n".join(lines)
