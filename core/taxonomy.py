from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd


TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "catalog" / "taxonomy.json"


@lru_cache(maxsize=1)
def load_taxonomy(path: Path | str = TAXONOMY_PATH) -> dict[str, dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def common_name(scientific_name: str) -> str:
    return load_taxonomy().get(scientific_name, {}).get(
        "common_name", scientific_name.replace("_", " ").title()
    )


def animal_group(scientific_name: str) -> str:
    return load_taxonomy().get(scientific_name, {}).get("group", "Other animals")


def enrich_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["common_name"] = enriched.species.map(common_name)
    enriched["animal_group"] = enriched.species.map(animal_group)
    return enriched
