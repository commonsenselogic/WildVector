from __future__ import annotations

from dataclasses import dataclass

from .scenario import WeatherScenario


@dataclass(frozen=True)
class ScenarioPreset:
    key: str
    label: str
    question: str
    description: str
    cause_label: str
    cause_value: str
    observation_prompt: str
    scenario: WeatherScenario


TYPICAL = ScenarioPreset(
    "typical",
    "🧭 Baseline",
    "Where do they travel each season?",
    "Show all recorded spring and fall journeys without reweighting them.",
    "Conditions",
    "Recorded baseline",
    "Notice what stays similar across years and what changes from animal to animal.",
    WeatherScenario(),
)


SPECIES_PRESETS = {
    "Cathartes aura": (
        TYPICAL,
        ScenarioPreset(
            "helpful-tailwind",
            "💨 Tailwind",
            "Can a tailwind change how vultures migrate?",
            "Add 12 km/h of route-aligned tailwind during measured flight hours.",
            "Flight wind",
            "+12 km/h tailwind",
            "Predict whether helpful wind changes their timing, speed, or need to rest.",
            WeatherScenario(tailwind_change_kmh=12.0),
        ),
        ScenarioPreset(
            "strong-headwind",
            "🌬️ Headwind",
            "How does a headwind change the trip?",
            "Subtract 12 km/h from route-aligned tailwind during measured flight hours.",
            "Flight wind",
            "12 km/h headwind",
            "Predict whether flying into the wind changes their timing, speed, or need to rest.",
            WeatherScenario(tailwind_change_kmh=-12.0),
        ),
    ),
    "Vulpes lagopus": (
        TYPICAL,
        ScenarioPreset(
            "warmer-spring",
            "☀️ +4 °C",
            "How could warmer migration seasons change fox movement?",
            "Raise air temperature by 4 °C inside both measured seasonal migration windows.",
            "Season temperature",
            "+4 °C warmer",
            "Predict how the same warmer condition changes spring and fall travel.",
            WeatherScenario(temperature_change_c=4.0),
        ),
        ScenarioPreset(
            "less-nearby-sea-ice",
            "🧊 Less sea ice",
            "How could less sea ice change fox movement?",
            "Lower sea-ice concentration by 25 percentage points and move the nearest valid ice 50 km farther away.",
            "Nearby sea ice",
            "25 points less and 50 km farther",
            "Predict whether less nearby ice changes the timing, pace, or path of seasonal movement.",
            WeatherScenario(
                sea_ice_concentration_change=-0.25,
                sea_ice_distance_change_km=50.0,
            ),
        ),
    ),
    "Balaenoptera musculus": (
        TYPICAL,
        ScenarioPreset(
            "stronger-surface-winds",
            "🌊 Windier",
            "Can stronger ocean winds change the whale route?",
            "Add 10 km/h to measured surface-wind conditions along spring and fall journeys.",
            "Ocean wind",
            "+10 km/h stronger",
            "Predict whether stronger surface winds change the whales' timing, pace, or path.",
            WeatherScenario(wind_speed_change_kmh=10.0),
        ),
        ScenarioPreset(
            "calmer-surface-winds",
            "🌤️ Calmer",
            "Can calmer ocean winds change the whale route?",
            "Subtract 8 km/h from measured surface-wind conditions along spring and fall journeys.",
            "Ocean wind",
            "8 km/h calmer",
            "Predict whether calmer surface winds change the whales' timing, pace, or path.",
            WeatherScenario(wind_speed_change_kmh=-8.0),
        ),
    ),
}


def presets_for_species(species: str) -> tuple[ScenarioPreset, ...]:
    return SPECIES_PRESETS.get(species, (TYPICAL,))
