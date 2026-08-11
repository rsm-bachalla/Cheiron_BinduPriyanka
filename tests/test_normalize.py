"""Normalization tests, driven by the shapes the live API actually returns."""

from app.normalize import (
    NO_PHASE_LABEL,
    normalize_studies,
    normalize_study,
    parse_start_year,
    phase_label,
)


def _raw(**overrides):
    protocol = {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A trial"},
        "statusModule": {
            "overallStatus": "RECRUITING",
            "startDateStruct": {"date": "2022-08-11", "type": "ACTUAL"},
        },
        "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL"},
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "Merck", "class": "INDUSTRY"}
        },
        "contactsLocationsModule": {"locations": [{"country": "France"}]},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "X"}]},
    }
    protocol.update(overrides)
    return {"protocolSection": protocol}


class TestParseStartYear:
    def test_handles_all_three_widths_the_api_emits(self):
        # The API returns "2022", "2022-08", and "2022-08-11" interchangeably.
        assert parse_start_year("2022") == 2022
        assert parse_start_year("2022-08") == 2022
        assert parse_start_year("2022-08-11") == 2022

    def test_returns_none_rather_than_raising_on_bad_input(self):
        assert parse_start_year(None) is None
        assert parse_start_year("") is None
        assert parse_start_year("unknown") is None

    def test_rejects_implausible_years(self):
        assert parse_start_year("0001-01-01") is None


class TestPhaseLabel:
    def test_missing_phases_are_distinct_from_not_applicable(self):
        # Observational studies omit `phases`; "NA" is a real interventional
        # value. Collapsing them would misreport both.
        assert phase_label([]) == NO_PHASE_LABEL
        assert phase_label(["NA"]) == "Not Applicable"
        assert phase_label([]) != phase_label(["NA"])

    def test_multi_phase_becomes_one_combined_label(self):
        assert phase_label(["PHASE1", "PHASE2"]) == "Phase 1/Phase 2"


class TestNormalizeStudy:
    def test_flattens_nested_payload(self):
        study = normalize_study(_raw())
        assert study is not None
        assert study.nct_id == "NCT00000001"
        assert study.lead_sponsor == "Merck"
        assert study.countries == ["France"]
        assert study.start_year == 2022
        assert study.url == "https://clinicaltrials.gov/study/NCT00000001"

    def test_dedupes_countries_across_sites(self):
        # A trial with many sites in one country must count once for it.
        raw = _raw(
            contactsLocationsModule={
                "locations": [
                    {"country": "United States", "city": "Boston"},
                    {"country": "United States", "city": "Nashville"},
                    {"country": "France", "city": "Paris"},
                ]
            }
        )
        assert normalize_study(raw).countries == ["United States", "France"]

    def test_survives_entirely_absent_modules(self):
        # Field projection returns only requested modules, and studies omit
        # modules that do not apply, so every module must be optional.
        study = normalize_study(
            {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}
        )
        assert study is not None
        assert study.phases == []
        assert study.countries == []
        assert study.start_year is None

    def test_drops_records_with_no_nct_id(self):
        # Without an ID a record cannot be cited, so it must not be counted.
        assert normalize_study({"protocolSection": {}}) is None
        assert normalize_studies([_raw(), {"protocolSection": {}}]) != []
        assert len(normalize_studies([_raw(), {"protocolSection": {}}])) == 1
