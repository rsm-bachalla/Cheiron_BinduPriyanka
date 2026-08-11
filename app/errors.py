"""Domain exceptions.

These carry enough structure that the API layer can turn them into a stable
error contract without re-deriving anything.
"""

from typing import Any


class AppError(Exception):
    """Base class for errors that map to a structured HTTP response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                **({"details": self.details} if self.details else {}),
            }
        }


class UnsupportedQueryError(AppError):
    """The query could not be mapped to a supported analysis with confidence.

    Raised instead of guessing. The caller gets an explicit statement of what
    was not understood plus what the service does support.
    """

    status_code = 422
    code = "unsupported_query"


class NoResultsError(AppError):
    """The query was understood, but ClinicalTrials.gov returned no studies."""

    status_code = 404
    code = "no_results"


class UnsupportedAnalysisError(AppError):
    """The query was understood, but that analysis is not implemented yet.

    Distinct from `unsupported_query`: the intent was mapped successfully, so
    the caller learns the request was valid and the capability is simply
    missing.
    """

    status_code = 501
    code = "analysis_not_implemented"


class UpstreamError(AppError):
    """ClinicalTrials.gov rejected the request or was unreachable."""

    status_code = 502
    code = "upstream_error"
