from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import joblib

from .catalog import DEFAULT_CATALOG_ROOT
from .classroom_catalog import CLASSROOM_POPULATIONS
from .classroom_scenarios import ScenarioPreset, presets_for_species
from .outcome_model import outcome_model_key, predict_activated_effects


SEASONS = ("spring migration", "fall migration")
EffectLoader = Callable[[str, str, ScenarioPreset], dict[str, dict]]


@dataclass(frozen=True)
class ScenarioSupport:
    species: str
    population: str
    scenario_key: str
    scenario_label: str
    supported: bool
    seasons: tuple[str, ...]
    outcomes: tuple[str, ...]


def catalog_effect_loader(
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> EffectLoader:
    root = Path(catalog_root)

    def load(species: str, population: str, preset: ScenarioPreset) -> dict[str, dict]:
        effects: dict[str, dict] = {}
        for season in SEASONS:
            key = outcome_model_key(species, population, season)
            path = root / "outcome-models" / f"{key}.joblib"
            if not path.exists():
                continue
            bundle = joblib.load(path)
            if int(bundle.get("schema_version", 0)) >= 2:
                effects[season] = predict_activated_effects(bundle, preset.scenario)
        return effects

    return load


def classroom_scenario_support(effect_loader: EffectLoader) -> tuple[ScenarioSupport, ...]:
    """Evaluate the exact non-baseline experiments exposed in the classroom."""
    reports: list[ScenarioSupport] = []
    for classroom_population in CLASSROOM_POPULATIONS:
        for preset in presets_for_species(classroom_population.species):
            if preset.scenario.is_typical:
                continue
            effects = effect_loader(
                classroom_population.species,
                classroom_population.population,
                preset,
            )
            active = {
                season: season_effects
                for season, season_effects in effects.items()
                if season_effects
            }
            reports.append(
                ScenarioSupport(
                    species=classroom_population.species,
                    population=classroom_population.population,
                    scenario_key=preset.key,
                    scenario_label=preset.label,
                    supported=bool(active),
                    seasons=tuple(sorted(active)),
                    outcomes=tuple(
                        sorted(
                            {
                                outcome
                                for season_effects in active.values()
                                for outcome in season_effects
                            }
                        )
                    ),
                )
            )
    return tuple(reports)


def validate_catalog_scenarios(
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> tuple[ScenarioSupport, ...]:
    return classroom_scenario_support(catalog_effect_loader(catalog_root))


def write_scenario_support_registry(
    reports: tuple[ScenarioSupport, ...],
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
) -> Path:
    root = Path(catalog_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "classroom-scenario-support.json"
    temporary = root / ".classroom-scenario-support.tmp.json"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rule": (
                    "Every exposed non-baseline preset must change at least one outcome "
                    "from a model that passed both forward-held-out baselines."
                ),
                "all_supported": all(report.supported for report in reports),
                "scenarios": [asdict(report) for report in reports],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
