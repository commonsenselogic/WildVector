import pandas as pd

from core.catalog import load_manifest
from core.classroom_catalog import (
    CLASSROOM_POPULATIONS,
    classroom_study_keys,
    filter_classroom_populations,
)


def test_initial_classroom_catalog_has_one_population_per_movement_domain():
    assert len(CLASSROOM_POPULATIONS) == 3
    assert {row.classroom_group for row in CLASSROOM_POPULATIONS} == {
        "Aquatic", "Terrestrial", "Avian",
    }
    assert len({row.species for row in CLASSROOM_POPULATIONS}) == 3


def test_classroom_studies_are_retained_in_expansion_manifest():
    manifest_keys = {study.key for study in load_manifest()}
    assert set(classroom_study_keys()) <= manifest_keys
    assert len(manifest_keys) == 30


def test_population_filter_uses_species_and_population_together():
    selected = CLASSROOM_POPULATIONS[0]
    frame = pd.DataFrame(
        [
            {"species": selected.species, "population": selected.population, "journeys": 10},
            {"species": selected.species, "population": "Different population", "journeys": 99},
            {"species": "Other species", "population": selected.population, "journeys": 99},
        ]
    )
    result = filter_classroom_populations(frame)
    assert len(result) == 1
    assert result.iloc[0].journeys == 10
