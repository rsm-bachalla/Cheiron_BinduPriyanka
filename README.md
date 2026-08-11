# Clinical Trials Insight API

An AI-enabled backend service that answers natural-language questions about
clinical trials using the ClinicalTrials.gov API and returns structured,
frontend-renderable visualization specifications with per-data-point source
citations.

## Design principle

**A deterministic analytics pipeline that an LLM configures but never executes.**

The LLM's entire job is to turn a question into one validated `QueryPlan`
object. Everything after that — filtering, counting, grouping, sorting, date
normalization, chart selection, citation attachment — is ordinary Python. The
LLM never sees study data, never counts anything, and never touches citations,
so it cannot fabricate a number or a source.

```
NL query
   │
   ▼
[1] Planner      ── LLM (structured output) or deterministic patterns ──▶ QueryPlan
   │                                                                       │
   │                          enum-validated; refuses rather than guessing
   ▼
[2] TrialsClient ── async httpx, Essie filters, paginated, field-projected ──▶ raw JSON
   │
   ▼
[3] Normalizer   ── flatten protocolSection ──▶ list[Study]
   │
   ▼
[4] Aggregator   ── deterministic counting ──▶ list[Bucket] (label, value, nct_ids)
   │
   ▼
[5] VizBuilder   ── operation + dimension ──▶ VisualizationSpec
   │
   ▼
[6] Citations    ── render NCT refs already carried by each bucket ──▶ QueryResponse
```

Stages 2–6 are pure functions over typed inputs and are individually testable.

## Quick start

```bash
make install
cp .env.example .env     # add your OPENAI_API_KEY
make run                 # http://localhost:8000/docs
```

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

**The service runs without credentials.** With no key it falls back to a
deterministic pattern planner that handles a narrow set of phrasings and refuses
everything else. The full test suite also runs with no key and no network.

## The planner

```
query → OpenAI structured output → Pydantic validation
      → semantic validation → merge hints → QueryPlan
```

The model returns a `QueryPlan` and nothing else — enum-constrained `operation`
and `dimension`, typed filters, comparison groups, a title. It never sees study
data, never counts, and never produces citations.

**Semantic validation is deterministic and lives outside the LLM.** Schema
validation proves a plan is well-formed; it cannot prove it is coherent. A
`time_trend` grouped by `sponsor` type-checks and is still nonsense. So
`plan_validation.py` independently enforces: trends group by year, geo by
country, network by sponsor; comparisons need ≥2 groups and no other operation
may carry them; year ranges must be ordered; a plan with no filters at all is
refused rather than scanning the whole registry.

**The failure ladder is bounded at one extra call:**

| Failure | Response |
| --- | --- |
| Schema mismatch or incoherent plan | One repair retry, naming the exact error |
| Auth / network / rate limit | No retry — re-prompting cannot help |
| Fallback + query matches a known pattern | Deterministic planner answers it |
| Fallback + ambiguous query | Structured `422` refusal |

**Hints override the model.** They are merged after validation, so
`{"query": "Show recruiting lung cancer trials", "hints": {"status": "COMPLETED"}}`
yields `COMPLETED`.

Swapping providers means writing one adapter satisfying the `LLMClient`
Protocol and adding a branch to `llm/factory.py`. Only OpenAI is implemented.

## API

### `POST /query`

```json
{
  "query": "How are breast cancer trials distributed across phases?",
  "hints": { "country": "France", "status": "RECRUITING" }
}
```

`query` is required. All `hints` are optional and **override** anything the
planner inferred, so a caller can always force a deterministic filter.

Response (abridged):

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
      "intent": "Count breast cancer trials grouped by phase.",
      "operation": "distribution",
      "dimension": "phase",
      "planner": "rulebased"
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

Rows are addressable through `encoding`, so a frontend can render any chart type
generically without special-casing the question that produced it.

### Comparisons

Each comparison group is its own ClinicalTrials.gov query, fanned out
concurrently with `asyncio.gather`, and every shared filter in the plan applies
to all of them. Group membership is decided by upstream matching — never by
fetching one broad result set and guessing locally who belongs where.

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

Three honesty rules follow from groups being independent queries:

- `meta.groups` reports `record_count`, `total_available`, and `truncated` **per
  group**. One series capped at 1000 while the other returns 258 is not a
  like-for-like chart, and the caller can see that.
- `meta.total_available` is **null** for comparisons. Group match sets overlap —
  a trial studying both drugs is legitimately in both series — so a combined
  total would be a number with no defensible meaning.
- **If one group fails upstream, the whole request fails** with a `502` naming
  the failed group. A chart missing one of its two series reads as a finding
  ("nivolumab has no phase 3 trials") when it is actually an outage.

### Networks

A relationship graph is not a list of rows, so it is not forced into one.
`network_graph` populates `nodes` and `edges` instead, and leaves `data` empty:

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

Nodes are lead sponsors and their drug interventions; an edge means the sponsor
ran at least one trial using that drug. **Edge weight is distinct trials, and a
trial contributes at most once to a given edge** no matter how many arms repeat
the intervention. Names are matched case-insensitively with whitespace
collapsed, so "Merck  Sharp & Dohme" and "merck sharp & dohme" are one node.

**Citations live on edges**, because the edge is the claim being made — and the
excerpt quotes both halves of it (`"Lead sponsor: … | Intervention: …"`), so a
reader can open the record and check the relationship rather than just the
existence of a trial.

Broad conditions produce thousands of pairs, so edges are capped at
`NETWORK_TOP_EDGES` by weight, with ties broken on name so the same input always
yields the same graph. Nodes orphaned by the cap are dropped. The cap, the
excluded studies, and the fact that node counts span the uncapped graph are all
disclosed in `meta.notes`.

One judgement call worth stating: **`BIOLOGICAL` interventions count as drugs.**
The registry uses the type interchangeably with `DRUG` for the same molecule —
across a 500-study melanoma sample, pembrolizumab was typed `DRUG` 29 times and
`BIOLOGICAL` 18 times. Accepting only `DRUG` splits one node in two and drops
roughly a third of the edges for exactly the monoclonal antibodies these
questions are usually about. The set is one constant in `network.py` and the
choice is named in the response notes.

### Errors

The service **refuses rather than guesses**. A question it cannot map with
confidence returns a structured `422` explaining why, instead of an answer to a
question you did not ask:

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

`501` is deliberately distinct from `422`: it tells the caller their question
was valid and the capability is simply missing, and echoes the interpreted
intent so that is verifiable.

## Source traceability

Every data point carries the NCT records that produced it. This is structural
rather than cosmetic: `nct_ids` are captured *during* aggregation, so a bucket
and its citations cannot disagree, and the citation `excerpt` quotes the exact
field value that placed the study in that bucket. A reader can open any cited
record and verify the claim.

## Query correctness

Parameter behaviour was validated against the live API *before* the client was
written. The headline finding: `query.intr=pembrolizumab` is only **84%**
precise (measured over 200 studies, 32 of which never mention the drug), while
`AREA[InterventionName]pembrolizumab` is **100%**. The client therefore uses
Essie `filter.advanced` expressions exclusively.

Full findings, including the silent `pageSize` cap at 1000 and the plain-text
error bodies: [`docs/api-findings.md`](docs/api-findings.md).

## Analytical honesty

`meta.notes` always discloses how numbers were produced:

- Multi-phase studies form their own bucket and are not double-counted, so
  bucket counts sum exactly to `record_count`.
- Multi-country studies are counted once per country, so geographic totals
  legitimately exceed the study count — stated explicitly.
- Studies with no usable start date are excluded from trends, and the excluded
  count is reported.
- When the record cap truncates, `truncated` is set and the note says the counts
  are a sample rather than registry totals.

## Tests

```bash
make test
```

Runs offline against stubbed upstream responses — no network, no LLM calls.

## Project layout

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
  viz.py              chart selection + spec construction
  citations.py        NCT references per data point
  llm/
    base.py           LLMClient Protocol + strict JSON-schema translation
    openai_client.py  OpenAI structured outputs over the shared httpx pool
    prompts.py        planner system prompt + few-shot examples
    factory.py        provider selection
  schemas/            plan.py (LLM contract), study.py, api.py (public contract)
docs/api-findings.md  verified upstream API behaviour
  network.py          sponsor-drug graph -> nodes + edges
tests/                151 tests, offline
```

## Status

**Working:** all five analyses — `distribution`, `time_trend`, `geo`,
`comparison`, and `network` — over phase, status, year, country, sponsor, and
sponsor type, via the OpenAI planner, with citations, disclosure notes, and the
full fallback ladder.

**Next:** the Streamlit demo, as a pure consumer of this HTTP contract.
