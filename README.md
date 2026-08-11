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

No credentials are required to run the service.

```bash
make install
make run          # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How are breast cancer trials distributed across phases?"}'
```

### Configuration

```bash
cp .env.example .env
```

The deterministic planner recognises a narrow set of phrasings and needs no
credentials. Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai` for open-ended
natural-language questions.

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
| `upstream_error` | 502 | ClinicalTrials.gov rejected the request or was unreachable |

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
  planner.py          NL -> QueryPlan (deterministic patterns; refuses when unsure)
  pipeline.py         orchestration
  clinicaltrials.py   async API client + Essie expression builder
  normalize.py        raw JSON -> Study
  aggregate.py        deterministic counting -> Bucket
  viz.py              chart selection + spec construction
  citations.py        NCT references per data point
  schemas/            plan.py (LLM contract), study.py, api.py (public contract)
docs/api-findings.md  verified upstream API behaviour
tests/
```

## Status

Implemented: deterministic vertical slice — distribution analyses over phase,
status, country, sponsor, sponsor type, and year, with citations and full
disclosure notes.

Next: OpenAI structured-output planner, then time-trend / comparison / network
analyses, then the Streamlit demo.
