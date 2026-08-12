# Clinical Trials Insight Agent

## 1. Overview

This service answers natural-language questions about clinical trials. A
question goes to OpenAI, which returns a validated plan describing what analysis
to run; the service then queries the ClinicalTrials.gov v2 API, normalizes the
records, and computes the answer in ordinary Python. The response is a
frontend-renderable visualization specification in which every data point carries
the NCT records that produced it, so any number on a chart can be traced back to
the registry entries behind it.

**The LLM decides what analysis to perform; deterministic Python executes the
analysis.**

The model never sees study data, never counts anything, and never produces a
citation, so it has no opportunity to fabricate a figure or a source. Everything
after the plan — filtering, grouping, counting, sorting, date handling, chart
selection, graph construction, citation attachment — is deterministic code with
unit tests.

## 2. Demo

```bash
make install
cp .env.example .env     # add your OPENAI_API_KEY
```

Two terminals:

```bash
# Terminal 1 — the API
make run

# Terminal 2 — the UI
source .venv/bin/activate
streamlit run demo/streamlit_app.py
```

- FastAPI docs: <http://localhost:8000/docs>
- Streamlit: <http://localhost:8501>

`make install` creates `.venv` but does not activate it, so activate first —
or run `make demo`, which is the same command through the virtualenv. Point the
demo at another host with `API_BASE_URL=https://host streamlit run
demo/streamlit_app.py`; it defaults to `http://localhost:8000` and shows a live
reachability indicator in the sidebar.

Five questions, one per supported analysis type — these are the buttons in the
demo:

| Analysis | Question |
| --- | --- |
| Distribution | How are breast cancer trials distributed across phases? |
| Time trend | How has the number of trials for pembrolizumab changed over time? |
| Geographic | Which countries have the most recruiting trials for lung cancer? |
| Comparison | Compare trial phases for pembrolizumab vs nivolumab |
| Network | Show a network of sponsors and drugs for melanoma trials |

Or without the UI:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How are breast cancer trials distributed across phases?"}'
```

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, or `rulebased` to disable the LLM |
| `OPENAI_API_KEY` | — | Required for open-ended questions |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any structured-output-capable model |
| `OPENAI_BASE_URL` | — | Set for an OpenAI-compatible gateway |
| `CTGOV_MAX_RECORDS` | `1000` | Per-query fetch cap; truncation is disclosed |
| `CITATIONS_PER_POINT` | `3` | Source records attached to each data point |
| `NETWORK_TOP_EDGES` | `40` | Sponsor-drug edges kept; the cap is disclosed |
| `API_BASE_URL` | `http://localhost:8000` | Where the Streamlit demo looks for this API |

**The service runs without credentials.** With no key it falls back to a
deterministic pattern planner that handles a narrow set of phrasings and refuses
everything else. The full test suite also runs with no key and no network.

## 3. Architecture

```
Streamlit                 demo/streamlit_app.py — HTTP client, imports nothing from app/
    ↓
FastAPI                   main.py — request contract, error contract, shared httpx pool
    ↓
OpenAI Query Planner      llm/ + planning.py — structured output, one repair retry
    ↓
Validated QueryPlan       schemas/plan.py + plan_validation.py — schema, then coherence
    ↓
ClinicalTrials.gov Client clinicaltrials.py — Essie filter.advanced, paginated
    ↓
Normalizer                normalize.py — raw protocolSection JSON -> list[Study]
    ↓
Deterministic Analytics   aggregate.py (buckets) / network.py (nodes + edges)
    ↓
Visualization Builder     viz.py — chart type + encoding from operation and dimension
    ↓
Citation Layer            citations.py — NCT refs from the IDs captured while counting
    ↓
Structured JSON Response  schemas/api.py — the public contract
```

- **Streamlit** is a demo client only. It calls `POST /query` over HTTP and
  imports no backend module.
- **FastAPI** owns the request/response contract and maps domain exceptions to
  status codes. One `httpx.AsyncClient` is created at startup and shared by both
  outbound integrations.
- **Query planner** turns the question into a `QueryPlan` — an enum-constrained
  operation, a dimension, typed filters, comparison groups, a title. Nothing else.
- **Validated QueryPlan** passes Pydantic schema validation and then a separate
  deterministic coherence check.
- **ClinicalTrials.gov client** builds Essie `filter.advanced` expressions,
  paginates by `nextPageToken`, and projects only the fields used.
- **Normalizer** flattens the nested API response into typed `Study` objects,
  treating every module as optional.
- **Analytics** counts. Bucket-shaped analyses go through `aggregate.py`; the
  relationship graph goes through `network.py`.
- **Visualization builder** picks the chart type and writes the `encoding` block
  that tells a frontend which key holds the category, the value, and the series.
- **Citation layer** renders NCT references from the IDs captured during
  aggregation, so a data point and its sources cannot disagree.

Explicitly:

- **OpenAI never computes trial counts.** It never receives study records.
- **OpenAI never constructs citations.** Citations are derived from NCT IDs
  captured while counting.
- **ClinicalTrials.gov is the source of truth.** Nothing is inferred about a
  trial that was not returned by the registry.
- **Analytics and network construction are deterministic.** Same records in,
  same numbers and same graph out, with ties broken on stable keys.

## 4. Supported analysis types

| Analysis | Example | Visualization |
| --- | --- | --- |
| Distribution | How are breast cancer trials distributed across phases? | `bar_chart` over phase, status, sponsor, or sponsor type |
| Time trend | How has the number of trials for pembrolizumab changed over time? | `line_chart` over start year |
| Geographic ranking | Which countries have the most recruiting trials for lung cancer? | `geo_ranking` — countries ordered by trial count |
| Comparison | Compare trial phases for pembrolizumab vs nivolumab | `grouped_bar_chart` with one series per group |
| Sponsor-drug network | Show a network of sponsors and drugs for melanoma trials | `network_graph` — `nodes` and `edges`, `data` empty |

Groupable dimensions: `phase`, `status`, `year`, `country`, `sponsor`,
`sponsor_type`. Filterable fields: drug, condition, sponsor, country, phase,
status, start year, end year.

## 5. API

### `GET /health`

```json
{ "status": "ok" }
```

Liveness only — it does not check OpenAI or ClinicalTrials.gov reachability.

### `POST /query`

```json
{
  "query": "How are breast cancer trials distributed across phases?",
  "hints": { "country": "France", "status": "RECRUITING" }
}
```

`query` is required. All `hints` are optional and **override** anything the
planner inferred, so a caller can always force a deterministic filter.

Response (abridged — one bucket of nine, one citation of three):

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
        "trial_count": 89,
        "citations": [
          {
            "nct_id": "NCT00002777",
            "url": "https://clinicaltrials.gov/study/NCT00002777",
            "title": "Exemestane Compared With Tamoxifen in Treating Women With ...",
            "excerpt": "Phase: PHASE3"
          }
        ]
      }
    ]
  },
  "meta": {
    "query_interpretation": {
      "intent": "Break down breast cancer trials by phase.",
      "operation": "distribution",
      "dimension": "phase",
      "planner": "openai"
    },
    "filters": { "condition": "breast cancer" },
    "source": "ClinicalTrials.gov",
    "record_count": 1000,
    "total_available": 16520,
    "truncated": true,
    "notes": ["Analysed 1000 of 16520 matching trials ..."]
  }
}
```

- **`visualization`** — the chart specification: a `type`, a human-readable
  `title`, an `encoding`, and the rows. Network responses populate `nodes` and
  `edges` instead of `data`.
- **`encoding`** — which key in each row is the category (`x`), the value (`y`),
  and, for comparisons, the series (`series`). A frontend reads this rather than
  hardcoding field names, so any analysis type renders generically. Fields that
  do not apply are omitted; the API is served with `response_model_exclude_none`,
  so absent means "not applicable", not `null`.
- **`data`** — one row per bucket, each carrying its own citations.
- **`citations`** — the NCT records behind that specific data point: ID, URL,
  title, and an `excerpt` quoting the field value that placed the study in that
  bucket. Capped at `CITATIONS_PER_POINT` per point.
- **`meta`** — how the answer was produced: the interpreted intent, the operation
  and dimension chosen, which planner ran, the filters actually applied, the
  source, record counts, and disclosure notes.
- **Truncation** — `record_count` is what was analysed and `total_available` is
  what matched upstream. When the fetch cap bites, `truncated` is `true` and a
  note states the counts are a sample rather than registry totals. For
  comparisons `total_available` is omitted and per-group figures appear in
  `meta.groups` (see §8).

### Errors

The service **refuses rather than guesses**:

```json
{
  "error": {
    "code": "unsupported_query",
    "message": "Could not determine what to group the trials by.",
    "details": {
      "reason": "no recognised grouping dimension (phase, status, country, sponsor, year)",
      "supported": "..."
    }
  }
}
```

| Code | Status | Meaning |
| --- | --- | --- |
| `unsupported_query` | 422 | Intent could not be mapped confidently |
| `no_results` | 404 | Understood, but no trials matched |
| `analysis_not_implemented` | 501 | Intent understood; that analysis is not built yet |
| `upstream_error` | 502 | ClinicalTrials.gov rejected the request or was unreachable |

`501` is deliberately distinct from `422`: it tells the caller their question was
valid and the capability is simply missing. It is **currently unreachable** —
all five operations in the plan schema are implemented, so nothing raises it
today. It is kept because the next operation added to the enum will need it
before its pipeline exists, and because collapsing it into `422` would tell a
caller their valid question was unintelligible.

## 6. Query planning

```
natural language
  → OpenAI structured output   (strict JSON schema; temperature 0)
  → Pydantic validation        (enums, types, ranges)
  → semantic validation        (is this plan coherent?)
  → hints merged last          (explicit caller filters override the model)
  → QueryPlan
```

The model returns a `QueryPlan` and nothing else. `operation` and `dimension` are
enum-constrained, filters are typed, and OpenAI's strict structured-output mode
enforces the schema on its side before the response is even parsed.

**Semantic validation is deterministic and lives outside the LLM.** Schema
validation proves a plan is well-formed; it cannot prove it is coherent. A
`time_trend` grouped by `sponsor` type-checks and is still nonsense. So
[plan_validation.py](app/plan_validation.py) independently enforces: trends group
by year, geo by country, network by sponsor; comparisons need ≥2 distinct groups
and no other operation may carry them; the comparison field must not also be
pinned in `filters`; year ranges must be ordered; a plan with no filters at all is
refused rather than scanning the whole registry.

**Hints override the model**, and are merged after validation, so
`{"query": "Show recruiting lung cancer trials", "hints": {"status": "COMPLETED"}}`
yields `COMPLETED`.

**The failure ladder is bounded at one extra call:**

| Failure | Response |
| --- | --- |
| Schema mismatch or incoherent plan | One repair retry, naming the exact error |
| Auth / network / rate limit | No retry — re-prompting cannot help |
| Fallback + query matches a known pattern | Deterministic planner answers it |
| Fallback + ambiguous query | Structured `422` refusal |

So there is one safe OpenAI flow with a single bounded repair; the rule-based
planner catches only obvious supported phrasings; and anything still ambiguous
returns a structured error rather than a guessed answer. Provider error bodies
never cross the boundary into client-facing details — only the status code does,
because OpenAI echoes a partially masked API key back on a 401.

Swapping providers means writing one adapter satisfying the `LLMClient` Protocol
and adding a branch to [llm/factory.py](app/llm/factory.py). Only OpenAI is
implemented.

## 7. ClinicalTrials.gov integration

Parameter behaviour was measured against the live API *before* the client was
written, because getting the query layer wrong produces confidently-wrong answers
that no amount of downstream correctness recovers. Full write-up:
[docs/api-findings.md](docs/api-findings.md).

### Upstream API findings

| Finding | Effect on implementation |
| --- | --- |
| **`query.*` is relevance search, not filtering.** `query.intr=pembrolizumab` was 84% precise over 200 studies — 32 never mentioned the drug, ranking instead on a title phrase like "a PD-1 inhibitor". | The `query.*` family is not used anywhere. |
| **`AREA[...]` via `filter.advanced` is precise.** The same drug filter as `AREA[InterventionName]"pembrolizumab"` was 100% precise on the same sample. | All filtering is Essie `filter.advanced`, clauses joined with ` AND `. Status is the exception and uses the native `filter.overallStatus`. |
| **`pageSize` silently caps at 1000.** `pageSize=1001` returns HTTP 200 with exactly 1000 studies and no warning. | Pagination is driven by `nextPageToken` with an explicit local record cap, never by trusting the requested page size. |
| **`totalCount` requires `countTotal=true`.** Omitted, the field is simply absent. | The client always sends it on the first page — truncation disclosure depends on knowing the true match count. |
| **Errors are plain text, not JSON.** `filter.overallStatus=BOGUS` returns `HTTP 400 Invalid value in parameter ...`. | Error handling never assumes a JSON body. Status and phase values are constrained by local enums so those cases are caught before the request is sent. |
| **Quotes are significant and unbalanced quotes are fatal.** Essie has no escape sequence, so an embedded `"` produces `HTTP 400 token recognition error`. | `_quote()` strips double quotes rather than escaping them, and free-text values from the model are sanitized before reaching the expression. This is an injection surface, not a formatting nicety. |
| **`phases` has three distinct states:** a phase list, `["NA"]`, or the field absent entirely (typically observational). | Absent and `NA` are kept as separate buckets — "Not Specified" vs "Not Applicable" — rather than collapsed. Genuinely multi-phase studies form their own combined bucket instead of incrementing each phase, so bucket counts sum exactly to the study count. |
| **`AREA[ConditionSearch]` is a curated search, broader than exact condition matching.** It returned 16,520 breast cancer studies against 14,997 for the stricter `AREA[Condition]`. | `ConditionSearch` is used deliberately: the stricter field drops genuinely relevant trials. The looseness is disclosed here and made auditable by citations. |

Two smaller ones: requested field-projection modules that hold no data come back
as `{}` rather than being omitted, so normalization treats every module as
optional; and `startDateStruct.date` occurs as `"2022"`, `"2022-08"`, and
`"2022-08-11"`, so year extraction takes the leading four characters.

## 8. Comparison design

Each comparison group is its own ClinicalTrials.gov request. The groups are
fanned out concurrently with `asyncio.gather`, every shared filter in the plan
applies to all of them, and results stay separate all the way to the response.
Group membership is decided by upstream matching — never by fetching one broad
result set and guessing locally who belongs where.

Rows carry a `group` and the encoding declares it as the series:

```json
{
  "type": "grouped_bar_chart",
  "encoding": {
    "x": { "field": "phase", "type": "nominal", "title": "Trial Phase" },
    "y": { "field": "trial_count", "type": "quantitative", "title": "Number of Trials" },
    "series": { "field": "group", "type": "nominal", "title": "Group" }
  },
  "data": [
    { "phase": "Phase 3", "group": "pembrolizumab", "trial_count": 114, "citations": [] },
    { "phase": "Phase 3", "group": "nivolumab", "trial_count": 82, "citations": [] }
  ]
}
```

Output is **dense**: every (label, group) pair is emitted, with an explicit zero
where a group has no trials, so a gap renders as a zero-height bar rather than a
missing row a frontend has to infer.

Three consequences of groups being independent queries:

- **Truncation is reported independently per group**, because it lands unevenly.
  `meta.groups` carries `record_count`, `total_available`, and `truncated` for
  each series. Comparing pembrolizumab with vorinostat, the first returns 1,000
  capped records of 2,528 matches while the second returns all 274 — one series
  is a sample and the other is complete, which is not a like-for-like chart. A
  single combined truncation flag would hide that; `meta.notes` also names the
  affected series specifically.
- **`total_available` is not summed.** Group match sets legitimately overlap — a
  trial studying both drugs really is in both series, and one was verified in the
  live registry — so a combined total would be a number with no defensible
  meaning. It is omitted for comparisons.
- **If one group fails upstream, the whole request fails** with a `502` naming
  the failed group. A chart missing one of its two series reads as a finding
  ("nivolumab has no phase 3 trials") when it is actually an outage.

## 9. Network design

A relationship graph is not a list of rows, so it is not forced into one.
`network_graph` populates `nodes` and `edges` and leaves `data` empty:

```json
{
  "type": "network_graph",
  "nodes": [
    { "id": "Bristol-Myers Squibb", "label": "Bristol-Myers Squibb",
      "node_type": "sponsor", "trial_count": 41 },
    { "id": "Ipilimumab", "label": "Ipilimumab", "node_type": "drug", "trial_count": 33 }
  ],
  "edges": [
    { "source": "Bristol-Myers Squibb", "target": "Ipilimumab",
      "trial_count": 16, "citations": [ ... ] }
  ]
}
```

- Nodes are **lead sponsors** and the **interventions** they run; an edge means
  the sponsor ran at least one trial using that intervention.
- Intervention types included: **`DRUG` and `BIOLOGICAL`** (see below).
  Procedures, devices, behavioural arms, and diagnostics are excluded, since
  including them turns a sponsor-drug map into a sponsor-everything map.
- **Edges are supported by unique NCT IDs**, and edge weight is the number of
  distinct trials.
- **One trial contributes at most once to a given sponsor-intervention edge**, no
  matter how many arms repeat the intervention.
- **High-cardinality results are capped deterministically** at
  `NETWORK_TOP_EDGES` by weight, with ties broken on name so the same input
  always yields the same graph. Nodes orphaned by the cap are dropped. The cap,
  the excluded studies, and the fact that node counts span the uncapped graph are
  all disclosed in `meta.notes`.
- **Citations attach to edges**, because the edge is the claim being made — and
  the excerpt quotes both halves of it (`"Lead sponsor: … | Intervention: …"`), so
  a reader can open the record and check the relationship rather than just the
  existence of a trial.

Names are matched case-insensitively with whitespace collapsed, so
`"Merck  Sharp & Dohme"` and `"merck sharp & dohme"` are one node.

**Why `BIOLOGICAL` counts as a drug.** The obvious rule is to accept only
`DRUG`. Inspecting the live registry showed that is wrong: the same molecule is
typed both ways across studies. Across a 500-study melanoma sample, pembrolizumab
was typed `DRUG` 29 times and `BIOLOGICAL` 18 times. Accepting only `DRUG` splits
one node in two and drops roughly a third of the edges — for exactly the
monoclonal antibodies these questions are usually about. Widening the set moved
Bristol-Myers Squibb → Ipilimumab from 6 shared trials to 16 and brought
Merck → Pembrolizumab into the top five. The set is one constant in
[network.py](app/network.py) and the choice is named in the response notes.

## 10. Reliability / correctness

- **Deterministic aggregation.** Every number is produced by Python over records
  returned by the registry; the LLM never counts.
- **Schema validation.** OpenAI strict structured outputs plus Pydantic on the
  way in, and a typed response model on the way out.
- **Semantic plan validation.** A separate deterministic coherence pass that a
  well-formed but nonsensical plan cannot pass.
- **Query sanitization.** Model-supplied free text is whitespace-collapsed and
  stripped of stray punctuation and quotes before it enters an Essie expression.
- **Truncation disclosure.** `truncated`, `total_available`, and an explicit note
  whenever counts are a sample rather than registry totals.
- **Safe provider error handling.** Upstream and LLM error bodies are classified,
  never echoed — only the status code reaches the client.
- **Structured unsupported-query errors.** Ambiguity produces a `422` explaining
  why, not an answer to a question the caller did not ask.
- **Source citations.** NCT IDs are captured *during* aggregation, so a bucket
  and its citations cannot disagree, and the excerpt quotes the exact field value
  that placed the study in that bucket.

`meta.notes` always discloses how numbers were produced: that multi-phase studies
form their own bucket and are not double-counted; that multi-country studies are
counted once per country, so geographic totals legitimately exceed the study
count; how many studies were excluded from a trend for having no usable start
date; and when the record cap truncated the analysis.

## 11. Testing

```bash
make test
```

**151 backend tests**, all offline — no network access and no API credentials
required.

| File | Tests | Covers |
| --- | --- | --- |
| [tests/test_planning.py](tests/test_planning.py) | 30 | Planner orchestration, semantic validation, hint merging, the failure ladder |
| [tests/test_network.py](tests/test_network.py) | 26 | Graph construction, dedupe, the top-N cap, node counts, HTTP round trip |
| [tests/test_comparison.py](tests/test_comparison.py) | 21 | Fan-out, shared filters, per-group truncation, partial-failure handling |
| [tests/test_openai_client.py](tests/test_openai_client.py) | 18 | Strict-schema translation, request contract, failure classification |
| [tests/test_planner.py](tests/test_planner.py) | 14 | Deterministic pattern planner, including its refusals |
| [tests/test_clinicaltrials.py](tests/test_clinicaltrials.py) | 12 | Essie expression building, pagination, quoting, upstream errors |
| [tests/test_aggregate.py](tests/test_aggregate.py) | 11 | Bucketing, multi-phase and multi-country semantics, notes |
| [tests/test_api.py](tests/test_api.py) | 10 | End-to-end HTTP contract and error responses |
| [tests/test_normalize.py](tests/test_normalize.py) | 9 | Raw JSON → `Study`, absent modules, date widths |

Both outbound integrations are stubbed at the HTTP boundary with
`httpx.MockTransport` — ClinicalTrials.gov and OpenAI alike — so tests exercise
the real client code including error paths, without a network.

**The Streamlit demo has no automated tests.** It was verified end-to-end with
Streamlit's `AppTest` runner, which executes the real page and surfaces
exceptions: all five query classes against a live backend, plus the
unsupported-query (`422`) and backend-unreachable paths. Those runs were manual,
not part of `make test`.

## 12. Important tradeoffs / limitations

- **Large result sets are sampled.** `CTGOV_MAX_RECORDS` caps the fetch at 1000
  records per query, so a broad question analyses a subset. This is disclosed on
  every affected response rather than hidden.
- **`AREA[ConditionSearch]` is broader than exact condition matching**, so a
  "breast cancer" query includes some adjacent studies. The stricter alternative
  drops genuinely relevant trials; citations make the tradeoff auditable.
- **The network graph is intentionally bounded** to the strongest
  `NETWORK_TOP_EDGES` relationships, and intervention names are used as recorded
  by the sponsor rather than resolved to a common vocabulary.
- **The Streamlit network view is static**, not draggable — Altair gives tooltips
  and zoom, not interactive layout.
- **No authentication**, no rate limiting.
- **Local demo only** — nothing is deployed.
- **No persistent datastore**; nothing is cached or stored between requests.
- **The LLM is used only for query interpretation.** It is not in the path of any
  number, chart, or citation.

## 13. AI tools used

Claude Code was used throughout implementation for scaffolding, implementation
assistance, test generation, debugging, and documentation. OpenAI is used at
runtime, for query planning only.

Generated and adapted code was validated rather than trusted:

- review of every generated file before commit
- deterministic unit tests, including deliberately adversarial cases
- stubbed upstream payloads, shaped after real API responses, replayed through
  `httpx.MockTransport` at both integration boundaries
- live ClinicalTrials.gov verification of query semantics, precision, paging, and
  error behaviour
- live OpenAI planner verification against real questions
- manual inspection of individual registry records — e.g. opening `NCT03715205`
  to confirm its sponsor and intervention exactly matched the citation the
  service emitted
- end-to-end Streamlit runs against the live backend

That process caught real defects. Two examples: adding a description to one enum
field made Pydantic emit a `$ref` with a sibling keyword, which OpenAI strict
mode rejects with an HTTP 400 — it broke every LLM plan and only surfaced because
a live query silently degraded to the fallback planner. And the obvious
`DRUG`-only intervention rule turned out to undercount by roughly a third once
measured against real data.

The following decisions were made deliberately by me, not delegated:

- **The deterministic analytics boundary** — what the LLM is allowed to decide
  versus what Python must compute.
- **API query semantics** — using Essie `filter.advanced` exclusively after
  measuring `query.*` precision.
- **Truncation behavior** — cap and disclose rather than silently sample or
  attempt unbounded pagination.
- **Comparison fan-out** — independent concurrent queries per group, and failing
  the whole request rather than returning a misleading partial chart.
- **Citation architecture** — capturing NCT IDs during aggregation so a data
  point and its sources are structurally inseparable.
- **Sponsor/drug network semantics** — what a node is, what an edge means, unique
  trials as edge weight, and the `BIOLOGICAL` inclusion.
- **Safe fallback behavior** — a bounded failure ladder that refuses rather than
  guesses.

## 14. Repository structure

```
app/
  main.py             FastAPI app, lifespan, error contract
  config.py           settings
  errors.py           domain exceptions -> structured HTTP responses
  planning.py         planner orchestration: LLM -> validate -> hints -> fallback
  planner.py          deterministic pattern planner (fallback; refuses when unsure)
  plan_validation.py  semantic coherence checks, independent of the LLM
  pipeline.py         plan -> fetch -> normalize -> aggregate -> visualize -> cite
  clinicaltrials.py   async API client + Essie expression builder
  normalize.py        raw JSON -> Study
  aggregate.py        deterministic counting -> Bucket (incl. grouped comparison)
  network.py          sponsor-drug graph -> nodes + edges
  viz.py              chart selection + spec construction
  citations.py        NCT references per data point
  llm/
    base.py           LLMClient Protocol + strict JSON-schema translation
    openai_client.py  OpenAI structured outputs over the shared httpx pool
    prompts.py        planner system prompt + few-shot examples
    factory.py        provider selection
  schemas/            plan.py (LLM contract), study.py, api.py (public contract)
demo/
  streamlit_app.py    Streamlit UI; an HTTP client, imports nothing from app/
docs/
  api-findings.md     verified upstream API behaviour, with measurements
tests/                151 tests, offline
PLAN.md               design decisions and their rationale, recorded as built
```

The demo renders each chart type with the lightest thing that works —
`st.bar_chart` and `st.line_chart` natively; Altair only where there is no native
equivalent (ordered horizontal bars for geo, `xOffset` for grouped comparison, a
bipartite diagram for the network). Altair and pandas ship with Streamlit, so no
plotting or graph library is added. Every renderer reads
`visualization.encoding` rather than hardcoding field names, which is what makes
the demo evidence that the contract is genuinely renderable.

## 15. Future improvements

- Pagination beyond the current analysis cap, or background processing for
  questions whose true match set is large.
- Caching of upstream responses and plans.
- Persistent query history.
- A richer visualization client — an interactive graph, drill-down from a bar to
  its trials.
- More analysis operations and dimensions.
- Production concerns: authentication, rate limiting, observability.
