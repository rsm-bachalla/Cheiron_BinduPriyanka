# Clinical Trials Insight API — Implementation Plan

An AI-enabled backend service that answers natural-language questions about clinical
trials using the ClinicalTrials.gov API and returns structured, frontend-renderable
visualization specifications.

**Status:** Steps 1–3 complete (deterministic pipeline + OpenAI planner). Steps 4–5 outstanding.
**Stack:** Python · FastAPI · Pydantic v2 · httpx (async) · pytest
**LLM provider:** OpenAI only (structured outputs). See §4.2.

---

## 1. Core thesis

The interesting engineering here is **a deterministic analytics pipeline that an LLM
configures but never executes.**

The LLM's entire job is to turn a natural-language question into one validated object —
a `QueryPlan`. Everything downstream is ordinary, testable Python. The LLM never sees
raw study data, never counts anything, and never touches citations.

That boundary is the single most important design decision in this project. It is what
separates a backend service from an LLM demo, and it maps directly onto the evaluation
weighting (System Design 35% + AI/Agent Design 20%).

---

## 2. Architecture

```
NL query
   │
   ▼
[1] Interpreter  ── LLM, structured output ──▶  QueryPlan (Pydantic, enum-validated)
   │                                                  │
   │                        hints merged in; LLM never sees raw study data
   ▼
[2] TrialsClient  ── async httpx, paginated, field-projected ──▶  raw JSON
   │
   ▼
[3] Normalizer   ── flatten protocolSection ──▶  list[Study]  (flat, typed)
   │
   ▼
[4] Aggregator   ── registry keyed by AnalysisOp ──▶  list[Bucket]  (label, value, nct_ids)
   │
   ▼
[5] VizBuilder   ── op → chart type + encoding ──▶  VisualizationSpec
   │
   ▼
[6] Citations    ── attach NCT refs + excerpts per data point ──▶  QueryResponse
```

Stages 2–6 are pure functions over typed inputs. Stage 1 is the only nondeterminism in
the system, and it is boxed in by schema validation.

### The LLM contract

One call, one object:

```python
class QueryPlan(BaseModel):
    intent: str                        # human-readable restatement, surfaced in meta
    operation: AnalysisOp              # TIME_TREND | DISTRIBUTION | COMPARISON | GEO | NETWORK
    dimension: Dimension               # PHASE | YEAR | COUNTRY | SPONSOR | STATUS | ...
    filters: TrialFilters              # drug, condition, phase, sponsor, country, status, years
    comparison_groups: list[str] = []  # ["pembrolizumab", "nivolumab"] for COMPARISON
    title: str
```

| Field | Example | Who decides |
| --- | --- | --- |
| `operation` | `DISTRIBUTION` | LLM (enum-validated) |
| `dimension` | `PHASE` | LLM (enum-validated) |
| `filters` | `condition="breast cancer"` | LLM, overridden by caller hints |
| `comparison_groups` | `["drug A", "drug B"]` | LLM |

Enums everywhere, so a hallucinated `"Phase 7"` fails validation rather than silently
corrupting a query.

**Planner flow (implemented):**

```
query → OpenAI structured output → Pydantic validation
      → semantic validation → merge hints → QueryPlan
```

**Failure ladder, bounded at one extra call:**

| Failure | Response |
| --- | --- |
| Schema/validation mismatch (repairable) | One repair retry, naming the exact error |
| Auth / network / rate limit (not repairable) | No retry — straight to fallback |
| Semantically incoherent plan | One repair, unless re-prompting cannot help |
| Fallback, query matches a supported pattern | Deterministic planner answers it |
| Fallback, query ambiguous | Structured `422` refusal |

The service does **not** guarantee an answer to every query. When intent cannot be
mapped confidently it returns a structured refusal rather than guessing — and the
refusal distinguishes "we cannot parse this" from "the provider was down".

**Hint precedence:** hints are merged *after* validation, so an explicit caller filter
always overrides the model's extraction.

**Semantic validation is deterministic and lives outside the LLM** (`plan_validation.py`):
`time_trend` must group by year, `geo` by country, `network` by sponsor; `comparison`
needs ≥2 groups and no other operation may carry them; year ranges must be ordered; a
plan with no filters at all is refused rather than scanning the registry.

**Free-text filter values are sanitised.** `drug`, `condition`, `sponsor`, and `country`
are the only unconstrained fields, so they are the one place prompt artifacts can reach
the upstream query — see §9.

---

## 3. Folder structure

```
app/
  main.py                 FastAPI app, lifespan (shared httpx client), error handlers
  config.py               pydantic-settings: provider, model, keys, page caps, timeouts
  api/
    routes.py             POST /query, GET /health
    errors.py             domain exceptions → HTTP problem responses
  schemas/
    request.py            QueryRequest (query + optional hints)
    plan.py               QueryPlan, TrialFilters, AnalysisOp, Dimension   ← LLM contract
    study.py              Study (normalized, flat)
    response.py           QueryResponse, VisualizationSpec, Encoding, DataPoint, Citation, Meta
  llm/
    base.py               LLMClient protocol: structured(schema, system, user) -> BaseModel
    anthropic_client.py   forced tool-use for schema adherence
    openai_client.py      OpenAI-compatible json_schema response_format
    rulebased_client.py   keyword/regex fallback — runs with zero credentials
    factory.py            provider selection from config
    prompts.py            system prompt + few-shot plan examples
  interpreter/
    planner.py            build plan, merge hints (hints win), validate + repair
  clients/
    clinicaltrials.py     async client: search(), pagination, retry/backoff, field projection
  analytics/
    normalize.py          raw JSON → Study; date, phase, country, sponsor extraction
    aggregate.py          op registry: time_trend, distribution, comparison, geo, network
    dates.py              partial-date handling ("2022-08" → 2022-08-01), year bucketing
  viz/
    selector.py           AnalysisOp (+ cardinality) → chart type
    builder.py            list[Bucket] → VisualizationSpec with encoding
  citations/
    attach.py             per-point NCT citations, URLs, excerpt selection
tests/
  fixtures/               captured real API payloads (recorded once, committed)
  test_normalize.py
  test_aggregate.py
  test_viz.py
  test_citations.py
  test_planner.py         with stub LLM client
  test_api.py             end-to-end against fixtures, no network
README.md
pyproject.toml
.env.example
Makefile
```

---

## 4. Major design decisions

**4.1 — The LLM emits a plan, not an answer.**
Single structured call. No agent loop, no tool-calling ping-pong, no framework. The
"agentic" surface is one schema-constrained generation with a validation/repair/fallback
ladder around it.

**4.2 — OpenAI only, behind a one-method Protocol. No LangChain / LangGraph.**
The entire agentic surface is one function call. A framework would add a dependency, an
indirection layer, and its own prompt templating for zero benefit. Provider abstraction
is a `Protocol` with one method:

```python
async def structured(schema: type[BaseModel], system: str, user: str, *,
                     repair_hint: str | None = None) -> BaseModel: ...
```

OpenAI implements it via `response_format={"type": "json_schema", "strict": true}`.
**Only OpenAI is implemented** — the assignment supplies an OpenAI key, and completing
and validating the product beats breadth of provider support. Another provider would be
one new adapter plus one branch in `llm/factory.py`; nothing else changes.

The adapter calls the REST endpoint over the shared `httpx` pool rather than the vendor
SDK: the surface used is a single POST, so going direct keeps dependencies small and
makes the boundary trivial to stub with `MockTransport`.

**Strict-mode schema translation.** OpenAI strict mode requires every object to mark all
properties required with `additionalProperties: false`, and rejects several keywords
Pydantic emits from `Field` constraints (`minimum`, `maximum`, …). `to_strict_schema()`
performs that translation; the dropped constraints are re-enforced by Pydantic on the way
back in and by semantic validation after that.

**4.3 — Every aggregator emits the same `Bucket` shape.**
`{label, value, group?, nct_ids: list[str]}` — including `network`, whose buckets are
edges. One uniform shape means one citation code path and one viz-building code path
instead of five. Adding a sixth analysis type later is a new function plus one registry
entry.

**4.4 — Citations are a byproduct of aggregation, not a bolted-on step.**
`nct_ids` ride inside the bucket from the moment it is created, so every data point
already knows which trials produced it. The citation layer only formats: top-K (default
3) per point, each with `nct_id`, `url`, `brief_title`, and an `excerpt` drawn from the
field that justified the bucket assignment. **The LLM never touches citations, so they
cannot be fabricated** — which is the entire point of the traceability bonus criterion.

**4.5 — Comparisons fan out, not filter down.**
"Drug A vs Drug B" becomes N independent API searches executed concurrently with
`asyncio.gather`, each tagged with its group label, then merged into a grouped series.
More correct than one broad query post-filtered locally, and it is where async genuinely
earns its place rather than being decorative.

**4.6 — Field projection at the client.**
Request only the ~12 study fields the pipeline actually reads via the `fields` parameter.
Smaller payloads, faster pagination, and it documents the real data dependency in code.

**4.7 — Truncation is always visible.**
`meta` reports both `record_count` (what we analysed) and the API's `totalCount` (what
exists). When the fetch cap truncates, a note says so explicitly. Silent partial answers
are worse than slow ones.

---

## 5. API contract

### Request

```json
{
  "query": "How are breast cancer trials distributed across phases?",
  "hints": {
    "condition": "breast cancer",
    "trial_phase": "Phase 3",
    "country": "United States",
    "status": "RECRUITING",
    "start_year": 2018,
    "end_year": 2024
  }
}
```

`query` is the only required field. All hints are optional; when present they **override**
the LLM's extracted filters rather than merging loosely, so a caller can always force a
deterministic filter.

### Response

```json
{
  "visualization": {
    "type": "bar_chart",
    "title": "Breast Cancer Trials by Phase",
    "encoding": {
      "x": { "field": "phase", "type": "nominal", "title": "Trial Phase" },
      "y": { "field": "trial_count", "type": "quantitative", "title": "Number of Trials" }
    },
    "data": [
      {
        "phase": "Phase 3",
        "trial_count": 41,
        "citations": [
          { "nct_id": "NCT01234567", "url": "https://clinicaltrials.gov/study/NCT01234567", "excerpt": "..." }
        ]
      }
    ]
  },
  "meta": {
    "query_interpretation": { "intent": "...", "operation": "DISTRIBUTION", "dimension": "PHASE" },
    "filters": { "condition": "breast cancer" },
    "source": "ClinicalTrials.gov",
    "record_count": 123,
    "total_available": 456,
    "notes": ["Trials with multiple phases are grouped as a distinct label, not double-counted."]
  }
}
```

---

## 6. Query & visualization coverage

| Analysis op | Example query | Chart type |
| --- | --- | --- |
| `TIME_TREND` | "How has the number of trials for pembrolizumab changed over time?" | `line_chart` |
| `DISTRIBUTION` | "How are breast cancer trials distributed across phases?" | `bar_chart` |
| `COMPARISON` | "Compare trial phases for Drug A vs Drug B." | `grouped_bar_chart` |
| `GEO` | "Which countries have the most recruiting trials for lung cancer?" | `choropleth` / `bar_chart` |
| `NETWORK` | "Show a network of sponsors and drugs for melanoma trials." | `network_graph` |

Chart selection is deterministic — a function of the operation plus result cardinality
(e.g. high-cardinality distributions degrade to a top-N bar chart with an "other" bucket).

---

## 7. Build order

Each step ends with something runnable.

| Step | Deliverable | Runnable at end of step |
| --- | --- | --- |
| **1. Scaffold** ✅ | `pyproject.toml`, config, folder structure, `.env.example`, `GET /health`. Deps verified on Python 3.14. | Health endpoint |
| **2. Deterministic spine** ✅ *(no LLM)* | Trials client → normalizer → `distribution` aggregator → viz builder → citations, wired to `POST /query`. | Breast-cancer-by-phase query with real citations, zero LLM involvement |
| **3. The agent** ✅ | OpenAI structured-output planner behind the Protocol; semantic validation; repair → fallback → refusal ladder; stub-client tests. | Arbitrary natural-language queries |
| **4. Coverage** | `comparison` (concurrent fan-out) and `network` (nodes + edges) aggregators. `time_trend` and `geo` are already live — they are counts over a different axis, so they reuse `distribution`. | All five example queries pass |
| **5. Tests + docs + Streamlit demo** | README architecture diagram, `demo/streamlit_app.py` as a pure HTTP consumer of the API contract. | Submission-ready |

Step 2 is deliberately LLM-free: the deterministic core is proven correct **before** any
nondeterminism enters the system, and there is a working demo early on a 24-hour clock.

Steps 1–4 constitute the submission. Step 5 is what makes it read as engineering rather
than a script.

---

## 8. Assumptions

- ClinicalTrials.gov API v2 remains public and unauthenticated. *(Verified live: `GET /api/v2/studies?query.intr=pembrolizumab&countTotal=true` → `200`, `totalCount: 2924`.)* Polite rate limiting, no key required.
- A per-query fetch cap (default ~1000 studies, configurable) bounds latency. `totalCount` is always reported alongside `record_count` so truncation is visible.
- Trials with multiple phases (`["PHASE1","PHASE2"]`) are labelled `"Phase 1/Phase 2"` as a distinct bucket rather than double-counted. Stated in `meta.notes`.
- A trial running in 5 countries counts once per distinct country in geo views, so the column total exceeds the trial count. Stated in `meta.notes`.
- "Over time" defaults to **study start year** unless the query clearly implies completion or first-posted date.

---

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| **Python 3.14.5** is newer than much of the ecosystem | Pydantic v2 / httpx / FastAPI ship 3.14 wheels. Pin conservatively and verify the install resolves as the *first* implementation step; flag rather than silently swap libraries. |
| **Essie query syntax** — ClinicalTrials.gov `query.*` params are not naive keyword fields; term stuffing returns junk | Build the param mapping against real responses and assert on it in tests. This is the real accuracy risk, more than the LLM. |
| **Sponsor/drug network explodes** | Cap to top-N sponsors × top-N interventions by trial count; state the cap in `meta.notes`. |
| **LLM emits an invalid or nonsensical plan** | Enum-constrained schema → validation → one repair retry → rule-based fallback. Never propagates to the query layer. |
| **Missing or partial start dates** | Handled in `normalize.py` for `YYYY`, `YYYY-MM`, `YYYY-MM-DD`; undated trials are excluded from time trends and counted in a `meta.notes` disclosure. |
| **LLM prompt artifacts leaking into filter values** *(observed live, now fixed)* | A model returned `pembrolizumab},` as the drug name. ClinicalTrials.gov tokenised the junk away and returned the **correct** studies, so the corrupted filter was invisible in the data and visible only in the echoed meta. Fixed on both sides: `TrialFilters` strips stray punctuation from all four free-text fields, and the few-shot examples are pretty-printed rather than inline JSON. Pinned by regression tests. |
| **Provider error bodies leaking to API clients** *(observed live, now fixed)* | OpenAI's 401 body echoes a partially masked API key, and it was reaching `error.details.llm_error`. Only the status code now crosses the boundary; the body is logged server-side. |

---

## 10. Local setup (target)

```bash
cp .env.example .env        # optional: add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
make install
make run                    # http://localhost:8000/docs

curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How are breast cancer trials distributed across phases?"}'
```

Runs with **no credentials** via the rule-based planner. Adding a key upgrades
interpretation quality without changing any other behaviour.
