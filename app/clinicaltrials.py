"""Async ClinicalTrials.gov API v2 client.

Parameter behaviour here was validated against the live API before this module
was written; see `docs/api-findings.md`. The two decisions that matter:

1. Filtering uses Essie `filter.advanced` expressions (`AREA[...]`), not the
   `query.*` parameters. Measured over 200 studies, `query.intr=pembrolizumab`
   returned 84% precision (32 studies did not list the drug at all), while
   `AREA[InterventionName]pembrolizumab` returned 100%.
2. `pageSize` is silently capped at 1000 by the server -- requesting 1001
   returns 1000 studies with HTTP 200 and no warning -- so pagination is driven
   explicitly by `nextPageToken` and a local record cap.
"""

import asyncio
import logging

import httpx

from app.config import Settings
from app.errors import UpstreamError
from app.schemas.plan import TrialFilters

logger = logging.getLogger(__name__)

# Only the fields the pipeline actually reads. Projection keeps payloads small
# and documents the real data dependency.
STUDY_FIELDS = ",".join(
    [
        "protocolSection.identificationModule.nctId",
        "protocolSection.identificationModule.briefTitle",
        "protocolSection.statusModule.overallStatus",
        "protocolSection.statusModule.startDateStruct",
        "protocolSection.designModule.phases",
        "protocolSection.designModule.studyType",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor",
        "protocolSection.contactsLocationsModule.locations",
        "protocolSection.armsInterventionsModule.interventions",
    ]
)

# Server-enforced ceiling; requesting more is silently truncated.
MAX_PAGE_SIZE = 1000


def _quote(value: str) -> str:
    """Render a value as an Essie phrase literal.

    Unbalanced double quotes make the server reject the whole expression with
    HTTP 400 ("token recognition error"), so embedded quotes are stripped rather
    than escaped -- Essie has no escape sequence for them.
    """
    cleaned = value.replace('"', " ").strip()
    cleaned = " ".join(cleaned.split())
    return f'"{cleaned}"'


def build_filter_expression(filters: TrialFilters) -> str | None:
    """Compose filters into a single Essie advanced-filter expression.

    Quoted phrase matching is deliberate: `AREA[ConditionSearch]"breast cancer"`
    matched 16,520 studies versus 16,697 unquoted, the difference being studies
    that match the words separately rather than as a phrase.
    """
    clauses: list[str] = []
    if filters.drug:
        clauses.append(f"AREA[InterventionName]{_quote(filters.drug)}")
    if filters.condition:
        clauses.append(f"AREA[ConditionSearch]{_quote(filters.condition)}")
    if filters.sponsor:
        clauses.append(f"AREA[LeadSponsorName]{_quote(filters.sponsor)}")
    if filters.country:
        clauses.append(f"AREA[LocationCountry]{_quote(filters.country)}")
    if filters.phase:
        clauses.append(f"AREA[Phase]{filters.phase.value}")
    if filters.start_year or filters.end_year:
        start = f"{filters.start_year}-01-01" if filters.start_year else "MIN"
        end = f"{filters.end_year}-12-31" if filters.end_year else "MAX"
        clauses.append(f"AREA[StartDate]RANGE[{start},{end}]")
    return " AND ".join(clauses) if clauses else None


class ClinicalTrialsClient:
    """Thin, paginating wrapper over `GET /api/v2/studies`."""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def search(
        self, filters: TrialFilters, max_records: int | None = None
    ) -> tuple[list[dict], int | None]:
        """Fetch studies matching `filters`.

        Returns `(raw_studies, total_available)`. `total_available` is the
        upstream match count, which exceeds `len(raw_studies)` when the local
        record cap truncates -- the caller discloses that in the response meta.
        """
        cap = max_records or self._settings.ctgov_max_records
        page_size = min(self._settings.ctgov_page_size, MAX_PAGE_SIZE, cap)

        params: dict[str, str | int] = {
            "fields": STUDY_FIELDS,
            "pageSize": page_size,
            "countTotal": "true",  # omitted -> totalCount absent from response
        }
        expression = build_filter_expression(filters)
        if expression:
            params["filter.advanced"] = expression
        if filters.status:
            # A native parameter rather than an AREA clause; invalid values are
            # rejected upstream with a plain-text 400.
            params["filter.overallStatus"] = filters.status.value

        studies: list[dict] = []
        total: int | None = None
        page_token: str | None = None

        while len(studies) < cap:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
                # countTotal is only meaningful on the first page.
                page_params.pop("countTotal", None)

            payload = await self._get(page_params)
            if total is None:
                total = payload.get("totalCount")

            batch = payload.get("studies", [])
            if not batch:
                break
            studies.extend(batch)

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return studies[:cap], total

    async def _get(self, params: dict) -> dict:
        """GET with bounded retries on transport errors and 5xx/429."""
        url = f"{self._settings.ctgov_base_url}/studies"
        last_error: Exception | None = None

        for attempt in range(self._settings.ctgov_max_retries):
            try:
                response = await self._client.get(
                    url, params=params, timeout=self._settings.ctgov_timeout_seconds
                )
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(2**attempt * 0.5)
                continue

            if response.status_code == 200:
                return response.json()

            # 4xx other than rate limiting is our bug (bad filter expression),
            # not a transient condition -- surface it immediately. The body is
            # plain text, not JSON.
            if response.status_code != 429 and response.status_code < 500:
                raise UpstreamError(
                    "ClinicalTrials.gov rejected the request.",
                    status=response.status_code,
                    upstream_message=response.text[:300],
                )

            last_error = UpstreamError(
                f"Upstream returned {response.status_code}",
                status=response.status_code,
            )
            await asyncio.sleep(2**attempt * 0.5)

        raise UpstreamError(
            "ClinicalTrials.gov is unavailable after retries.",
            cause=str(last_error) if last_error else None,
        )
