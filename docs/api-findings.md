# ClinicalTrials.gov API v2 — verified behaviour

Measured against the live API before the client was written, because getting the
query layer wrong produces confidently-wrong answers that no amount of downstream
correctness recovers. Every claim below was checked, not assumed.

Base URL: `https://clinicaltrials.gov/api/v2/studies` — public, no authentication.

## 1. `query.*` parameters are relevance search, not filters

This is the single most consequential finding.

Sampling 200 studies per method and checking whether the drug actually appears in
the study's own intervention list:

| Method | Precision |
| --- | --- |
| `query.intr=pembrolizumab` | **84%** (168/200) |
| `filter.advanced=AREA[InterventionName]pembrolizumab` | **100%** (199/200) |

`query.intr` matched studies whose interventions were Mavrostobart, Tislelizumab,
Gemcitabine and Docetaxel — they ranked because the title mentions "a PD-1
inhibitor". For counting and charting that is simply wrong data.

**Decision:** all filtering uses Essie `filter.advanced` expressions. The
`query.*` family is not used anywhere.

## 2. Essie `AREA[...]` fields in use

All verified to return HTTP 200 and correctly-filtered results:

| Filter | Expression | Spot-check |
| --- | --- | --- |
| Drug | `AREA[InterventionName]"pembrolizumab"` | 100% precision (above) |
| Condition | `AREA[ConditionSearch]"breast cancer"` | 16,520 studies |
| Sponsor | `AREA[LeadSponsorName]"Merck"` | 2,730 studies |
| Country | `AREA[LocationCountry]"France"` | 50/50 sampled really had a France site |
| Phase | `AREA[Phase]PHASE3` | 49,629 studies |
| Start date | `AREA[StartDate]RANGE[2020-01-01,2021-12-31]` | all sampled starts fell in range |

Clauses combine with ` AND `. `RANGE` accepts `MIN` / `MAX` sentinels for
open-ended bounds.

Status is the exception: it uses the native `filter.overallStatus` parameter
rather than an `AREA` clause.

## 3. Quoting is significant, and unbalanced quotes are fatal

`AREA[ConditionSearch]"breast cancer"` → 16,520 studies (phrase match)
`AREA[ConditionSearch]breast cancer` → 16,697 studies (looser, words may match separately)

We quote for accuracy. But a value containing a double quote breaks the whole
expression:

```
AREA[ConditionSearch]"bad"quote"
→ HTTP 400  Error parsing query in advanced filter: token recognition error at: '"'
```

Essie has no escape sequence for embedded quotes, so `_quote()` strips them
rather than escaping. This is an injection surface into the filter expression,
not merely a formatting nicety.

## 4. `pageSize` is silently capped at 1000

`pageSize=1001` returns HTTP 200 with exactly 1000 studies and no warning. Code
that trusts the requested size would silently under-read. Pagination is driven
by `nextPageToken` with an explicit local record cap.

Page tokens round-trip correctly with no overlap between consecutive pages
(verified on a 5-per-page melanoma query).

## 5. `totalCount` requires `countTotal=true`

Omitted, the field is simply absent from the response. The client always sends
it, because truncation disclosure depends on knowing the true match count.

## 6. Errors are plain text, not JSON

```
filter.overallStatus=BOGUS
→ HTTP 400  Invalid value in parameter `overallStatus`: `BOGUS`
```

Error handling must not assume a JSON body. Status and phase values are
constrained by local enums so these are caught before the request is made.

## 7. `phases` has three distinct states

From a 200-study breast cancer sample:

| Value | Count | Meaning |
| --- | --- | --- |
| `["PHASE2"]` etc. | 103 | Single phase |
| `["NA"]` | 53 | Interventional, phase not applicable |
| *field absent* | 43 | Typically observational |
| `["PHASE1","PHASE2"]` | 7 | Genuinely multi-phase |

Absent and `"NA"` are **not** the same thing and are kept as separate buckets
("Not Specified" vs "Not Applicable"). Multi-phase studies form their own
combined bucket rather than incrementing each phase, so bucket counts always sum
to the study count.

## 8. Field projection accepts dotted paths

`fields=protocolSection.identificationModule.nctId,...` works. Requested modules
that hold no data come back as `{}` rather than being omitted, so every module
must be treated as optional during normalization.

## 9. Dates come in three widths

`"2022"`, `"2022-08"`, `"2022-08-11"` all occur in `startDateStruct.date`. Year
extraction takes the leading four characters, which is well-defined for all three.

## Known limitation

`AREA[ConditionSearch]` is a curated condition search, not an exact-match filter.
A "breast cancer" query legitimately returns some studies whose primary subject
is adjacent (e.g. a pharmacokinetics study in patients that includes a breast
cancer cohort). This is upstream search semantics rather than a client defect;
the alternative, `AREA[Condition]`, is stricter (14,997 vs 16,520) but drops
genuinely relevant trials. Citations make the tradeoff auditable — a reader can
open any cited NCT record and judge it.
