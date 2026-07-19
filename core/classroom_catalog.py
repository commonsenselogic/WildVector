from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClassroomPopulation:
    species: str
    population: str
    study_key: str
    classroom_group: str
    movement_label: str


# The source manifest remains a 30-study expansion archive. These three populations
# are the initial classroom product because they combine repeated years, many long
# journeys, and strong environmental-data coverage for their movement domain.
CLASSROOM_POPULATIONS = (
    ClassroomPopulation(
        species="Cathartes aura",
        population="Eastern North American flyway",
        study_key="north_american_vultures",
        classroom_group="Avian",
        movement_label="seasonal migration",
    ),
    ClassroomPopulation(
        species="Balaenoptera musculus",
        population="Northeast Pacific",
        study_key="northeast_pacific_blue_whale",
        classroom_group="Aquatic",
        movement_label="seasonal migration",
    ),
    ClassroomPopulation(
        species="Vulpes lagopus",
        population="Canadian High Arctic",
        study_key="bylot_arctic_fox",
        classroom_group="Terrestrial",
        # This population is partially migratory; long journeys also include nomadism
        # and dispersal, so the classroom must not call every track a migration.
        movement_label="seasonal movement and long-distance dispersal",
    ),
)


def classroom_study_keys() -> tuple[str, ...]:
    return tuple(population.study_key for population in CLASSROOM_POPULATIONS)


def filter_classroom_populations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keys = pd.DataFrame(
        [
            {
                "species": population.species,
                "population": population.population,
                "classroom_group": population.classroom_group,
                "movement_label": population.movement_label,
            }
            for population in CLASSROOM_POPULATIONS
        ]
    )
    return frame.merge(keys, on=["species", "population"], how="inner", validate="one_to_one")
