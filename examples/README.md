# Example outputs

Five real responses from `POST /query`, one per supported analysis type. Each
file is the complete raw JSON the running backend returned, re-serialized with
indentation and otherwise unmodified — no values were edited, trimmed, or
hand-written. All five calls returned HTTP 200, every file parses with
`json.load()`, and every file validates against the service's own response model
(`app.schemas.api.QueryResponse`).

Generated on 2026-08-11 against the live ClinicalTrials.gov v2 API, with OpenAI
as the planner (`meta.query_interpretation.planner` is `openai` in all five).
Counts reflect the registry on that date and will drift as trials are
registered and updated.

| # | Query | File | Analysis | What it demonstrates |
| --- | --- | --- | --- | --- |
| 1 | How are breast cancer trials distributed across phases? | [01_distribution.json](01_distribution.json) | `distribution` → `bar_chart` | Nine phase buckets in which multi-phase studies (`Phase 1/Phase 2`) form their own bucket rather than incrementing both, and `Not Applicable` stays distinct from `Not Specified`, so bucket counts sum exactly to `record_count`. |
| 2 | How has the number of trials for pembrolizumab changed over time? | [02_time_trend.json](02_time_trend.json) | `time_trend` → `line_chart` | A 2010–2027 start-year series, showing that studies with unusable start dates are excluded and disclosed rather than silently dropped, and that future-dated registrations are reported as they appear upstream. |
| 3 | Which countries have the most recruiting trials for lung cancer? | [03_geographic.json](03_geographic.json) | `geo` → `geo_ranking` | 60 countries ranked by trial count, with a note stating that multi-country studies are counted once per country — so the column total legitimately exceeds the number of studies analysed. |
| 4 | Compare trial phases for pembrolizumab vs nivolumab | [04_comparison.json](04_comparison.json) | `comparison` → `grouped_bar_chart` | Two independent upstream queries kept as separate series, with per-group truncation in `meta.groups` and `total_available` deliberately omitted because the two cohorts can legitimately contain the same trial. |
| 5 | Show a network of sponsors and drugs for melanoma trials | [05_network.json](05_network.json) | `network` → `network_graph` | A sponsor–drug graph as `nodes` and `edges` with `data` empty, where edge weight is distinct supporting trials, citations attach to edges rather than rows, and the 40-of-1,283 edge cap is disclosed. |

## What is in each response

Every file carries the same envelope: a `visualization` (chart `type`, `title`,
an `encoding` block naming which row key is the category, the value, and the
series, and the rows themselves) and a `meta` block (interpreted intent, the
operation and dimension chosen, which planner ran, the filters actually applied,
record counts, and disclosure notes).

**Citations are per data point, not per response.** Each row — or each edge, in
the network — carries up to three NCT records with an `excerpt` quoting the
exact field value that placed the study in that bucket. Example 1 contains 27
distinct NCT IDs, example 5 contains 82.

## Truncation

Four of the five hit the 1,000-record fetch cap, which is disclosed in
`meta.truncated` and in `meta.notes` rather than left for the reader to infer.

| File | `record_count` | `total_available` | `truncated` |
| --- | --- | --- | --- |
| `01_distribution.json` | 1,000 | 16,520 | `true` |
| `02_time_trend.json` | 1,000 | 2,528 | `true` |
| `03_geographic.json` | 1,000 | 2,151 | `true` |
| `04_comparison.json` | 2,000 | omitted | `true` |
| `05_network.json` | 1,000 | 3,744 | `true` |

`04_comparison.json` is the interesting case: `total_available` is **absent**,
not null, because the two group match sets overlap and a combined total would
have no defensible meaning. Per-group figures live in `meta.groups` instead —
pembrolizumab 1,000 of 2,528 and nivolumab 1,000 of 1,663, both truncated. The
service is served with `response_model_exclude_none`, so an absent field means
"not applicable" rather than "null".

`05_network.json` carries a second, independent cap: 40 of 1,283 sponsor–drug
relationships were kept, ranked by shared-trial count with ties broken on name.
Node `trial_count` values span the uncapped graph, so they exceed the sum of the
edges shown — which is stated in `meta.notes` rather than left to surprise a
reader.

## Reproducing these

```bash
make run   # in one terminal

curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How are breast cancer trials distributed across phases?"}'
```

Counts will differ from these files as the registry changes. The structure,
the disclosure notes, and the citation behaviour will not.
