"""FastAPI application: routes, lifespan, and the error contract."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.clinicaltrials import ClinicalTrialsClient
from app.config import get_settings
from app.errors import AppError
from app.llm.factory import build_llm_client
from app.pipeline import run_query
from app.planning import plan_query
from app.schemas.api import QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One connection pool for the process; creating a client per request would
    # discard connection reuse across the many pages a single query fetches.
    settings = get_settings()
    async with httpx.AsyncClient(
        headers={"Accept": "application/json", "User-Agent": "clinical-trials-insight/0.1"},
        timeout=settings.ctgov_timeout_seconds,
    ) as http_client:
        app.state.http_client = http_client
        app.state.settings = settings
        # None when no usable credentials are configured, in which case the
        # service runs on the deterministic planner alone.
        app.state.llm_client = build_llm_client(settings, http_client)
        yield


app = FastAPI(
    title="Clinical Trials Insight API",
    description=(
        "Answers natural-language questions about clinical trials using the "
        "ClinicalTrials.gov API and returns frontend-renderable visualization "
        "specifications with per-data-point source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Domain errors become a stable, structured error body.

    The service refuses rather than guessing, so this path is a designed
    outcome and not merely a failure case.
    """
    logger.info("Returning %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Answer one natural-language question about clinical trials."""
    settings = request.app.state.settings
    client = ClinicalTrialsClient(request.app.state.http_client, settings)

    # Raises UnsupportedQueryError (HTTP 422) when intent cannot be mapped
    # confidently, rather than inventing an interpretation.
    outcome = await plan_query(
        payload.query, payload.hints, llm=request.app.state.llm_client
    )

    return await run_query(
        query=payload.query,
        hints=payload.hints,
        plan=outcome.plan,
        planner_name=outcome.planner,
        client=client,
        settings=settings,
    )
