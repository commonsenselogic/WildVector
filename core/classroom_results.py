from __future__ import annotations

from datetime import datetime, timedelta


OUTCOME_LABELS = {
    "departure_date": "Departure",
    "arrival_date": "Arrival",
    "migration_duration": "Travel time",
    "migration_pace": "Travel speed",
    "stopovers": "Rest stops",
    "corridor_choice": "Route choice",
}


def _format_date(day_of_year: float) -> str:
    day = max(1, min(366, int(round(day_of_year))))
    return (
        datetime(2020, 1, 1) + timedelta(days=day - 1)
    ).strftime("%b %d").replace(" 0", " ")


def _format_value(outcome: str, value: float, unit: str) -> str:
    if outcome in {"departure_date", "arrival_date"}:
        return _format_date(value)
    if outcome == "stopovers":
        return f"{value:.1f} stops"
    if unit:
        return f"{value:.1f} {unit}"
    return f"{value:.1f}"


def _continuous_sentence(outcome: str, delta: float, unit: str) -> str:
    amount = abs(delta)
    if outcome == "departure_date":
        return f"Departure is {amount:.1f} days {'later' if delta > 0 else 'earlier'}."
    if outcome == "arrival_date":
        return f"Arrival is {amount:.1f} days {'later' if delta > 0 else 'earlier'}."
    if outcome == "migration_duration":
        return f"The trip is {amount:.1f} days {'longer' if delta > 0 else 'shorter'}."
    if outcome == "migration_pace":
        return f"Travel is {amount:.1f} km/day {'faster' if delta > 0 else 'slower'}."
    if outcome == "stopovers":
        return f"The model predicts {amount:.1f} {'more' if delta > 0 else 'fewer'} rest stops."
    return f"The model changes this outcome by {delta:+.1f} {unit}."


def _change_short(outcome: str, delta: float, unit: str) -> str:
    amount = abs(delta)
    if outcome in {"departure_date", "arrival_date"}:
        return f"{amount:.1f} days {'later' if delta > 0 else 'earlier'}"
    if outcome == "migration_duration":
        return f"{amount:.1f} days {'longer' if delta > 0 else 'shorter'}"
    if outcome == "migration_pace":
        return f"{amount:.1f} km/day {'faster' if delta > 0 else 'slower'}"
    if outcome == "stopovers":
        return f"{amount:.1f} {'more' if delta > 0 else 'fewer'}"
    return f"{delta:+.1f} {unit}".strip()


def _corridor_name(label: str) -> str:
    return {
        "corridor 1": "western route",
        "corridor 2": "eastern route",
    }.get(str(label), str(label))


def _comparison_widths(baseline: float, scenario: float) -> tuple[float, float]:
    """Scale a pair against its larger value without inventing a target."""
    ceiling = max(abs(baseline), abs(scenario), 1e-9)
    return (
        max(4.0, min(100.0, abs(baseline) / ceiling * 100.0)),
        max(4.0, min(100.0, abs(scenario) / ceiling * 100.0)),
    )


def build_classroom_effect_rows(effects_by_season: dict) -> list[dict]:
    rows: list[dict] = []
    for season, effects in effects_by_season.items():
        for outcome, effect in effects.items():
            if effect.get("kind") == "continuous":
                baseline = float(effect["baseline"])
                scenario = float(effect["scenario"])
                delta = float(effect["delta"])
                unit = str(effect.get("unit", ""))
                before_width, after_width = _comparison_widths(baseline, scenario)
                rows.append(
                    {
                        "season": season,
                        "outcome": outcome,
                        "kind": "date" if outcome in {"departure_date", "arrival_date"} else "quantity",
                        "label": OUTCOME_LABELS.get(outcome, outcome.replace("_", " ").title()),
                        "before": _format_value(outcome, baseline, unit),
                        "after": _format_value(outcome, scenario, unit),
                        "baseline_value": baseline,
                        "scenario_value": scenario,
                        "unit": unit,
                        "before_width": before_width,
                        "after_width": after_width,
                        "change": delta,
                        "change_short": _change_short(outcome, delta, unit),
                        "sentence": _continuous_sentence(outcome, delta, unit),
                        "support": effect.get("support", "validated"),
                        "evidence_season": effect.get("evidence_season", season),
                    }
                )
                continue
            changes = {
                label: float(effect["scenario"].get(label, 0.0))
                - float(effect["baseline"].get(label, 0.0))
                for label in effect.get("baseline", {})
            }
            if not changes:
                continue
            label, delta = max(changes.items(), key=lambda item: abs(item[1]))
            route = _corridor_name(label)
            baseline = float(effect["baseline"][label])
            scenario = float(effect["scenario"][label])
            before_width = max(0.0, min(100.0, baseline * 100.0))
            after_width = max(0.0, min(100.0, scenario * 100.0))
            rows.append(
                {
                    "season": season,
                    "outcome": outcome,
                    "kind": "probability",
                    "label": OUTCOME_LABELS["corridor_choice"],
                    "before": f"{route.title()} {baseline:.0%}",
                    "after": f"{route.title()} {scenario:.0%}",
                    "baseline_value": baseline,
                    "scenario_value": scenario,
                    "unit": "probability",
                    "before_width": before_width,
                    "after_width": after_width,
                    "change": delta,
                    "change_short": f"{abs(delta) * 100:.0f} points {'more' if delta > 0 else 'less'}",
                    "sentence": (
                        f"The {route} becomes {abs(delta) * 100:.0f} percentage points "
                        f"{'more' if delta > 0 else 'less'} likely."
                    ),
                    "support": effect.get("support", "validated"),
                    "evidence_season": effect.get("evidence_season", season),
                }
            )
    return rows
