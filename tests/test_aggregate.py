"""Aggregation tests. These pin the counting invariants the response depends on."""

from app.aggregate import distribution
from app.schemas.plan import Dimension
from app.schemas.study import Study


def _study(nct: str, **kwargs) -> Study:
    return Study(nct_id=nct, brief_title=f"Trial {nct}", **kwargs)


class TestDistributionByPhase:
    def test_counts_sum_to_study_count(self):
        # The core invariant: multi-phase studies form their own bucket, so no
        # study is counted twice and the totals reconcile.
        studies = [
            _study("NCT1", phases=["PHASE1"]),
            _study("NCT2", phases=["PHASE1", "PHASE2"]),
            _study("NCT3", phases=["PHASE2"]),
            _study("NCT4", phases=[]),
        ]
        result = distribution(studies, Dimension.PHASE)
        assert sum(b.value for b in result.buckets) == len(studies)

    def test_multi_phase_does_not_increment_each_phase(self):
        result = distribution([_study("NCT1", phases=["PHASE1", "PHASE2"])], Dimension.PHASE)
        labels = {b.label: b.value for b in result.buckets}
        assert labels == {"Phase 1/Phase 2": 1}

    def test_buckets_carry_the_ids_that_produced_them(self):
        # Citations depend entirely on this; if IDs drift, citations lie.
        studies = [_study("NCT1", phases=["PHASE3"]), _study("NCT2", phases=["PHASE3"])]
        bucket = distribution(studies, Dimension.PHASE).buckets[0]
        assert bucket.value == len(bucket.nct_ids) == 2
        assert set(bucket.nct_ids) == {"NCT1", "NCT2"}

    def test_ordered_by_trial_progression_not_count(self):
        studies = [
            _study("NCT1", phases=["PHASE3"]),
            _study("NCT2", phases=["PHASE1"]),
            _study("NCT3", phases=["PHASE1"]),
        ]
        labels = [b.label for b in distribution(studies, Dimension.PHASE).buckets]
        assert labels == ["Phase 1", "Phase 3"]

    def test_notes_disclose_multi_phase_and_unspecified_handling(self):
        studies = [
            _study("NCT1", phases=["PHASE1", "PHASE2"]),
            _study("NCT2", phases=[]),
        ]
        notes = " ".join(distribution(studies, Dimension.PHASE).notes)
        assert "not counted under each phase" in notes
        assert "no phase recorded" in notes


class TestDistributionByCountry:
    def test_multi_country_study_counts_once_per_country(self):
        studies = [_study("NCT1", countries=["France", "Spain"])]
        result = distribution(studies, Dimension.COUNTRY)
        assert {b.label: b.value for b in result.buckets} == {"France": 1, "Spain": 1}

    def test_double_counting_is_disclosed(self):
        # The column total legitimately exceeds the study count here, so the
        # response must say so rather than let a reader mis-add.
        result = distribution([_study("NCT1", countries=["France"])], Dimension.COUNTRY)
        assert any("once per country" in n for n in result.notes)

    def test_ranked_by_count_descending(self):
        studies = [
            _study("NCT1", countries=["Spain"]),
            _study("NCT2", countries=["France"]),
            _study("NCT3", countries=["France"]),
        ]
        labels = [b.label for b in distribution(studies, Dimension.COUNTRY).buckets]
        assert labels == ["France", "Spain"]


class TestDistributionByYear:
    def test_undated_studies_excluded_and_disclosed(self):
        studies = [_study("NCT1", start_year=2020), _study("NCT2", start_year=None)]
        result = distribution(studies, Dimension.YEAR)
        assert sum(b.value for b in result.buckets) == 1
        assert any("no usable start date" in n for n in result.notes)

    def test_sorted_chronologically(self):
        studies = [_study("NCT1", start_year=2022), _study("NCT2", start_year=2019)]
        labels = [b.label for b in distribution(studies, Dimension.YEAR).buckets]
        assert labels == ["2019", "2022"]


class TestTopN:
    def test_truncates_and_discloses(self):
        studies = [_study(f"NCT{i}", countries=[f"C{i}"]) for i in range(10)]
        result = distribution(studies, Dimension.COUNTRY, top_n=3)
        assert len(result.buckets) == 3
        assert any("top 3 of 10" in n for n in result.notes)
