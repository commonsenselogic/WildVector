from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from core.catalog import CatalogError, CatalogStore, DEFAULT_CATALOG_ROOT, load_manifest
from core.classroom_scenarios import presets_for_species
from core.classroom_results import build_classroom_effect_rows
from core.classroom_catalog import filter_classroom_populations
from core.outcome_model import outcome_model_key, predict_activated_effects
from core.population import simulate_population_scenario
from core.scenario import WeatherScenario
from core.taxonomy import enrich_taxonomy
from core.visualization import population_route_geometry, render_population_deck
from core.migration_player import advance_migration_progress


st.set_page_config(page_title="Wild Vector", page_icon="🌎", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width:1600px;padding-top:.35rem;padding-bottom:1.4rem;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stAppViewBlockContainer"] {padding-left:2.25rem;padding-right:2.25rem;}
    .app-header {display:block;min-height:0;padding:.25rem 0 .65rem;
      border-bottom:1px solid rgba(35,205,209,.28);margin-bottom:.75rem;}
    .app-header h1 {font-size:2.15rem;line-height:1.02;margin:.08rem 0 .18rem;font-weight:800;
      letter-spacing:.015em;background:linear-gradient(90deg,#7bdc35,#22bff3);-webkit-background-clip:text;
      color:transparent;}
    .app-header p {margin:0;opacity:.72;font-size:.93rem;}
    .app-kicker,.eyebrow {font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;
      font-weight:700;opacity:.66;}
    .trust-pill {display:flex;align-items:center;gap:.45rem;white-space:nowrap;padding:.42rem .7rem;
      border:1px solid rgba(97,181,139,.42);border-radius:999px;background:rgba(60,173,115,.10);
      font-size:.78rem;font-weight:650;}
    .trust-dot {width:.5rem;height:.5rem;border-radius:50%;background:#45c486;
      box-shadow:0 0 0 4px rgba(69,196,134,.14);}
    .section-heading {display:flex;align-items:center;gap:.65rem;margin:.1rem 0 .55rem;}
    .step-number {display:grid;place-items:center;flex:0 0 1.7rem;height:1.7rem;border-radius:50%;
      background:linear-gradient(135deg,#72dc35,#16bde8);color:#06233d;font-size:.78rem;font-weight:800;}
    .section-heading strong {display:block;font-size:1.02rem;line-height:1.15;}
    .section-heading span:not(.step-number) {display:block;font-size:.74rem;opacity:.62;margin-top:.08rem;}
    .animal-card {background:linear-gradient(140deg,#e8f8ff,#eefbe8);color:#132c38;
      border:1px solid #b9dfe3;border-radius:16px;padding:9px 11px;margin:.35rem 0 .55rem;
      box-shadow:0 10px 30px rgba(26,65,81,.09);}
    .animal-card h2,.animal-card p {color:#132c38 !important;margin:0;}
    .animal-card h2 {font-size:1.08rem;line-height:1.15;margin-bottom:.15rem;white-space:nowrap;}
    .animal-card .population {font-size:.76rem;opacity:.78;margin-bottom:.25rem;}
    .animal-card .animal-meta {font-size:.68rem;font-weight:650;opacity:.82;}
    .mission-card {background:linear-gradient(135deg,#162b46,#214f69);color:#fff;
      border-radius:18px;padding:13px 14px;margin:.45rem 0 .5rem;
      box-shadow:0 14px 35px rgba(6,22,39,.20);}
    .mission-card h2,.mission-card p {color:#fff !important;margin:0;}
    .mission-card h2 {font-size:1.18rem;line-height:1.28;margin:.28rem 0 .55rem;}
    .mission-card .kicker {font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;
      opacity:.7;font-weight:700;}
    .cause-readout {display:flex;align-items:flex-start;gap:.55rem;padding:.55rem .65rem;
      border-radius:12px;background:rgba(255,255,255,.1);}
    .cause-readout b,.cause-readout span {display:block;color:#fff;}
    .cause-readout b {font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;opacity:.68;}
    .cause-readout span {font-size:.9rem;font-weight:650;margin-top:.05rem;}
    .look-for {font-size:.82rem;line-height:1.4;opacity:.82;margin-top:.55rem !important;}
    .map-heading {display:flex;align-items:flex-end;justify-content:space-between;gap:.8rem;
      margin:.08rem 0 .45rem;}
    .map-heading h2 {font-size:1.35rem;margin:0;}
    .map-heading p {font-size:.77rem;margin:.08rem 0 0;opacity:.62;}
    .data-badge {white-space:nowrap;font-size:.7rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.07em;padding:.28rem .52rem;border-radius:999px;border:1px solid rgba(120,150,170,.3);}
    .quest-key {display:flex;flex-wrap:wrap;align-items:center;gap:13px;margin:.1rem 0 .4rem;
      font-size:.78rem;font-weight:650;}
    .timeline-labels {display:flex;justify-content:space-between;font-size:.68rem;opacity:.62;
      margin-top:-.6rem;}
    .map-status {display:flex;justify-content:space-between;align-items:center;gap:.8rem;
      font-size:.75rem;margin:.28rem 0 0;opacity:.76;}
    .map-status b {font-weight:700;opacity:1;}
    .prediction-panel {background:linear-gradient(145deg,#08284a,#07566b);color:#f5fbff;
      border:1px solid rgba(62,222,211,.55);border-radius:18px;padding:14px 15px;margin:.15rem 0 .65rem;
      box-shadow:0 12px 30px rgba(0,106,151,.18);}
    .prediction-panel h2,.prediction-panel p {color:#f5fbff !important;margin:0;}
    .prediction-panel h2 {font-size:1.48rem;line-height:1.25;margin:.35rem 0 .5rem;}
    .prediction-panel .kicker {font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
      font-weight:750;color:#9fea5c;}
    .prediction-panel p {font-size:.88rem;line-height:1.45;}
    .comparison-card {background:#fff;color:#172934;border:1px solid #d8e1e6;border-radius:15px;
      padding:11px 12px;margin-bottom:.5rem;box-shadow:0 6px 18px rgba(20,42,62,.08);}
    .comparison-title {display:flex;align-items:center;justify-content:space-between;gap:.5rem;
      color:#65727b;font-size:.69rem;font-weight:750;text-transform:uppercase;letter-spacing:.08em;}
    .comparison-title .difference {color:#a34428;background:#fff0e9;border-radius:999px;
      padding:.16rem .4rem;letter-spacing:0;text-transform:none;white-space:nowrap;}
    .compare-grid {display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);
      align-items:center;gap:.42rem;margin:.55rem 0 .45rem;}
    .series-label {display:flex;align-items:center;gap:.28rem;font-size:.65rem;color:#65727b;
      font-weight:700;margin-bottom:.15rem;}
    .series-dot {display:inline-block;width:.48rem;height:.48rem;border-radius:50%;flex:0 0 .48rem;}
    .series-dot.baseline,.compare-fill.baseline {background:#4191cd;}
    .series-dot.scenario,.compare-fill.scenario {background:#a45cff;}
    .compare-value {font-size:.9rem;font-weight:800;line-height:1.2;min-height:2.15em;}
    .compare-track {height:.34rem;border-radius:999px;background:#e7edf0;overflow:hidden;margin-top:.3rem;}
    .compare-fill {height:100%;border-radius:999px;}
    .date-spacer {height:.34rem;margin-top:.3rem;}
    .compare-arrow {font-size:1.05rem;color:#84919a;font-weight:700;}
    .change-callout {border-top:1px solid #e8edef;padding-top:.42rem;color:#45545e;
      font-size:.76rem;font-weight:650;line-height:1.35;}
    .season-pair {display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:.55rem;}
    .season-effect {padding:.55rem;border-radius:11px;background:#f7fafb;border:1px solid #e2e9ed;}
    .season-heading {display:flex;align-items:center;justify-content:space-between;gap:.4rem;
      margin-bottom:.42rem;font-size:.76rem;font-weight:800;}
    .season-heading small {font-size:.6rem;font-weight:700;color:#687781;text-transform:uppercase;
      letter-spacing:.04em;}
    .season-effect .compare-grid {margin:.25rem 0 .4rem;}
    .season-effect .compare-value {font-size:.82rem;min-height:0;}
    .season-effect.fall .series-dot.baseline,.season-effect.fall .compare-fill.baseline {background:#2da578;}
    .season-effect.spring .series-dot.scenario,.season-effect.spring .compare-fill.scenario {background:#a45cff;}
    .season-effect.fall .series-dot.scenario,.season-effect.fall .compare-fill.scenario {background:#ffd54a;}
    .evidence-check {display:flex;gap:.65rem;padding:.7rem .75rem;border-radius:14px;
      background:rgba(57,174,112,.11);border:1px solid rgba(57,174,112,.28);margin:.5rem 0;}
    .check-icon {display:grid;place-items:center;flex:0 0 1.45rem;height:1.45rem;border-radius:50%;
      background:#35a86d;color:#fff;font-size:.78rem;font-weight:800;}
    .evidence-check b,.evidence-check span {display:block;}
    .evidence-check b {font-size:.82rem;}
    .evidence-check span {font-size:.73rem;opacity:.72;line-height:1.35;margin-top:.08rem;}
    .think-card {border-left:3px solid #5b9bd5;padding:.55rem .7rem;margin-top:.65rem;
      background:rgba(71,138,194,.09);border-radius:0 12px 12px 0;}
    .think-card b {display:block;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;
      margin-bottom:.16rem;}
    .think-card span {display:block;font-size:.83rem;line-height:1.38;}
    div[data-testid="stPills"] button {font-size:.84rem;}
    div[data-testid="stButton"] button {font-weight:700;}
    button[data-testid="stBaseButton-pillsActive"] {border-color:#26cfd4 !important;
      color:#8fe957 !important;background:rgba(21,185,211,.10) !important;}
    button[data-testid="stBaseButton-primary"] {border:0 !important;color:#06233d !important;
      background:linear-gradient(110deg,#7cde35,#20c4eb) !important;
      box-shadow:0 8px 22px rgba(16,189,213,.22) !important;}
    @media (max-width:900px) {
      [data-testid="stAppViewBlockContainer"] {padding-left:1rem;padding-right:1rem;}
      .app-header {display:block;}
    }
    @media (max-width:700px) {
      .app-header h1{font-size:1.65rem}.prediction-panel,.mission-card{padding:12px 13px;}
      .map-status{align-items:flex-start;flex-direction:column;gap:.2rem;}
      .season-pair{grid-template-columns:1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CATALOG_DATABASE = DEFAULT_CATALOG_ROOT / "wildvector.duckdb"
MAX_CLASSROOM_JOURNEYS = 36
GROUP_ICONS = {
    "Birds": "🐦",
    "Aquatic animals": "🐋",
    "Land mammals": "🦌",
    "Mammals": "🦇",
    "Land reptiles": "🐢",
}
SPECIES_ICONS = {
    "Cathartes aura": "🦅",
    "Vulpes lagopus": "🦊",
    "Balaenoptera musculus": "🐋",
}
STUDENT_GROUP_LABELS = {
    "Birds": "Bird",
    "Aquatic animals": "Aquatic animal",
    "Land mammals": "Land mammal",
    "Mammals": "Mammal",
    "Land reptiles": "Land reptile",
}


@st.cache_data(ttl=60, show_spinner=False)
def population_options(database_timestamp: float):
    del database_timestamp
    populations = CatalogStore(CATALOG_DATABASE).migration_populations()
    return enrich_taxonomy(filter_classroom_populations(populations))


@st.cache_data(ttl=3600, show_spinner=False)
def load_population(species: str, population: str):
    return CatalogStore(CATALOG_DATABASE).migration_telemetry(
        species, population, max_journeys=MAX_CLASSROOM_JOURNEYS
    )


@st.cache_resource(show_spinner=False)
def load_outcome_bundle(path_text: str, modified: float):
    del modified
    return joblib.load(path_text)


@st.cache_data(show_spinner=False, max_entries=48)
def run_cached_scenario(
    telemetry: pd.DataFrame,
    scenario: WeatherScenario,
    movement_type: str,
    scenario_effects: dict | None,
):
    return simulate_population_scenario(
        telemetry,
        scenario,
        movement_type,
        bins=36,
        activated_effects=scenario_effects,
    )


@st.cache_data(show_spinner=False, max_entries=48)
def cached_route_geometry(population_scenario, focus_seasons: tuple, scenario_selected: bool):
    """Cache the migration player's routing/animation geometry across full reruns.

    `run_cached_scenario` already returns a fresh copy of its result on every call, so
    switching species and switching back can't reuse `core.visualization`'s in-process,
    identity-keyed geometry cache -- it never sees the same object twice. Caching here,
    keyed on `population_scenario`'s content instead of identity, means revisiting a
    species/scenario combination skips the land/water routing pipeline entirely.
    """
    return population_route_geometry(population_scenario, list(focus_seasons), scenario_selected)


def model_validations(species: str, population: str) -> list[dict]:
    validations = []
    for season in ("spring migration", "fall migration"):
        key = outcome_model_key(species, population, season)
        path = DEFAULT_CATALOG_ROOT / "outcome-models" / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) >= 2:
                validations.extend(payload.get("validations", []))
    return validations


def model_source_trials(species: str, population: str) -> list[dict]:
    trials = []
    for season in ("spring migration", "fall migration"):
        key = outcome_model_key(species, population, season)
        path = DEFAULT_CATALOG_ROOT / "outcome-models" / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) >= 3:
                trials.extend(payload.get("source_trials", []))
    return trials


def activated_effects(species: str, population: str, scenario: WeatherScenario):
    effects = {}
    for season in ("spring migration", "fall migration"):
        key = outcome_model_key(species, population, season)
        path = DEFAULT_CATALOG_ROOT / "outcome-models" / f"{key}.joblib"
        if path.exists():
            bundle = load_outcome_bundle(str(path), path.stat().st_mtime)
            if int(bundle.get("schema_version", 0)) >= 2:
                effects[season] = predict_activated_effects(bundle, scenario)
    return effects


@st.cache_resource(show_spinner="Warming up every animal and experiment for instant switching...")
def warm_migration_cache(catalog_combinations: pd.DataFrame):
    """Precompute every species/population/preset's routing geometry once at launch.

    `load_population`, `run_cached_scenario`, and `cached_route_geometry` are all
    process-wide Streamlit caches, so warming them here (once, guarded by
    `st.cache_resource`) means every student's *first* pick of a species or experiment
    hits a warm cache instead of paying for telemetry loading and land/water routing
    on the spot.
    """
    for row in catalog_combinations.itertuples():
        telemetry = load_population(row.species, row.population)
        if telemetry.empty:
            continue
        for preset in presets_for_species(row.species):
            try:
                scenario = preset.scenario
                effects = activated_effects(row.species, row.population, scenario)
                result = run_cached_scenario(telemetry, scenario, row.movement_type, effects)
                cached_route_geometry(result, (), not scenario.is_typical)
            except Exception:
                # A single incomplete model/catalog combination shouldn't block the
                # rest of the warm-up; the normal page flow will surface the error
                # (see the "activated model response" check below) if a student picks it.
                continue
    return True


def section_heading(step: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <span class="step-number">{step}</span>
          <div><strong>{title}</strong><span>{subtitle}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def natural_join(items: list[str]) -> str:
    if len(items) < 2:
        return items[0] if items else ""
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


st.markdown(
    """
    <div class="app-header">
      <div>
        <h1>WILD VECTOR</h1>
        <p>Mapping nature. Understanding tomorrow.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

manifest = load_manifest()
store = CatalogStore(CATALOG_DATABASE)
if not store.available:
    st.warning("A teacher needs to build the local migration catalog before the quest can begin.")
    st.code(
        ".\\.venv\\Scripts\\python.exe scripts\\refresh_catalog.py --continue-on-error\n"
        ".\\.venv\\Scripts\\python.exe scripts\\refresh_environment.py --route-bins 6\n"
        ".\\.venv\\Scripts\\python.exe scripts\\train_population_models.py",
        language="powershell",
    )
    st.stop()

try:
    options = population_options(CATALOG_DATABASE.stat().st_mtime)
except CatalogError as exc:
    st.error(str(exc))
    st.stop()
if options.empty:
    st.warning("No population has enough long spring and fall journeys yet.")
    st.stop()

options = options.sort_values(["animal_group", "common_name", "population"]).reset_index(drop=True)
warm_migration_cache(
    options[["species", "population", "movement_type"]].drop_duplicates().reset_index(drop=True)
)
options["choice"] = options.apply(
    lambda row: (
        f"{SPECIES_ICONS.get(str(row.species), GROUP_ICONS.get(row.animal_group, '🐾'))} "
        f"{row.common_name} — {row.population}"
    ),
    axis=1,
)
default_matches = options.index[options.common_name.eq("Blue whale")].tolist()
default_index = default_matches[0] if default_matches else 0

control_column, map_column, result_column = st.columns(
    [1.0, 2.15, 1.25], gap="large"
)
with control_column:
    section_heading("1", "Choose an animal", "Three long-distance journeys")
    selected_choice = st.selectbox(
        "Choose your animal",
        options.choice.tolist(),
        index=default_index,
        label_visibility="collapsed",
    )
selected = options[options.choice.eq(selected_choice)].iloc[0]
selected_species = str(selected.species)
selected_population = str(selected.population)
selected_common_name = str(selected.common_name)
movement_type = str(selected.movement_type)

telemetry = load_population(selected_species, selected_population)
if telemetry.empty:
    st.error("This migration could not be loaded from the local catalog.")
    st.stop()

candidate_presets = presets_for_species(selected_species)
available_presets = list(candidate_presets)

with control_column:
    st.markdown(
        f"""
        <div class="animal-card">
          <h2>{SPECIES_ICONS.get(selected_species, GROUP_ICONS.get(str(selected.animal_group), '🐾'))} {selected_common_name}</h2>
          <p class="population">{selected_population}</p>
          <p class="animal-meta">{STUDENT_GROUP_LABELS.get(str(selected.animal_group), str(selected.animal_group))} · {int(selected.journeys)} journeys · {int(selected.years)} years</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_heading("2", "Change one condition", "Compare it with the baseline")
    preset_labels = [preset.label for preset in available_presets]
    selected_preset_label = st.pills(
        "Migration experiment",
        preset_labels,
        default=preset_labels[0],
        key=f"experiment_{selected_species}",
        label_visibility="collapsed",
    ) or preset_labels[0]
selected_preset = next(
    preset for preset in available_presets if preset.label == selected_preset_label
)
with control_column:
    st.markdown(
        f"""
        <div class="mission-card">
          <div class="kicker">Predict before you play</div>
          <h2>{selected_preset.question}</h2>
          <div class="cause-readout">
            <span>↗</span>
            <div><b>{selected_preset.cause_label}</b><span>{selected_preset.cause_value}</span></div>
          </div>
          <p class="look-for">{selected_preset.observation_prompt}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
scenario = selected_preset.scenario
historical_effects = activated_effects(selected_species, selected_population, scenario)
if not scenario.is_typical and not any(historical_effects.values()):
    st.error(
        "This classroom build is incomplete: a promised experiment has no activated "
        "model response. A teacher should retrain the population models before launch."
    )
    st.code(
        ".\\.venv\\Scripts\\python.exe scripts\\train_population_models.py\n"
        ".\\.venv\\Scripts\\python.exe scripts\\validate_classroom_scenarios.py",
        language="powershell",
    )
    st.stop()
validations = model_validations(selected_species, selected_population)
source_trials = model_source_trials(selected_species, selected_population)
result = run_cached_scenario(telemetry, scenario, movement_type, historical_effects)
historical_effects = result.activated_effects
active_outcome_names = sorted(
    {outcome for effects in historical_effects.values() for outcome in effects}
)
has_route_effect = "corridor_choice" in active_outcome_names
has_stopover_effect = "stopovers" in active_outcome_names
displayed_journeys = int(result.baseline.journeys.journey_id.nunique())
effect_rows = build_classroom_effect_rows(historical_effects)

with result_column:
    section_heading(
        "3",
        "Compare the impact",
        "Recorded baseline beside the model what-if" if not scenario.is_typical else "Start with recorded variation",
    )
    if scenario.is_typical:
        st.markdown(
            f"""
            <div class="prediction-panel">
              <div class="kicker">Recorded baseline</div>
              <h2>Journeys share a corridor—but never match exactly.</h2>
              <p>This map samples {displayed_journeys} recorded journeys from {result.baseline.animals} animals across {result.baseline.years} years.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif effect_rows:
        outcome_count = len({row["outcome"] for row in effect_rows})
        prediction_text = (
            "The scenario changes both seasonal journeys."
            if outcome_count == 1
            else f"The scenario changes {outcome_count} outcomes in both seasons."
        )
        season_text = "Spring + Fall"
        if has_route_effect:
            prediction_detail = (
                "Compare Spring and Fall together. Each scenario route keeps the known "
                "seasonal start and end while changing the journey between them."
            )
        elif has_stopover_effect:
            prediction_detail = (
                "The tested change is the number of rest stops. Purple and yellow show the "
                "corresponding evidence-weighted journeys while the paired cards show the counts."
            )
        else:
            prediction_detail = "Compare the paired Spring and Fall outcomes below."
        st.markdown(
            f"""
            <div class="prediction-panel">
              <div class="kicker">Model prediction · {season_text}</div>
              <h2>{prediction_text}</h2>
              <p>{prediction_detail}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rows_by_outcome = {}
        for row in effect_rows:
            rows_by_outcome.setdefault(row["outcome"], []).append(row)
        for rows in rows_by_outcome.values():
            season_cells = []
            for season in ("spring migration", "fall migration"):
                row = next((item for item in rows if item["season"] == season), None)
                if row is None:
                    continue
                before_bar = '<div class="date-spacer"></div>'
                after_bar = '<div class="date-spacer"></div>'
                if row["kind"] != "date":
                    before_bar = (
                        '<div class="compare-track"><div class="compare-fill baseline" '
                        f'style="width:{row["before_width"]:.1f}%"></div></div>'
                    )
                    after_bar = (
                        '<div class="compare-track"><div class="compare-fill scenario" '
                        f'style="width:{row["after_width"]:.1f}%"></div></div>'
                    )
                season_name = season.replace(" migration", "").title()
                season_class = season.split()[0]
                support_label = (
                    "Tested history"
                    if row["support"] == "validated"
                    else "Paired estimate"
                )
                season_cells.append(
                    f'<div class="season-effect {season_class}">'
                    f'<div class="season-heading"><span>{season_name}</span><small>{support_label}</small></div>'
                    '<div class="compare-grid"><div>'
                    '<div class="series-label"><span class="series-dot baseline"></span>Baseline</div>'
                    f'<div class="compare-value">{row["before"]}</div>{before_bar}</div>'
                    '<div class="compare-arrow">→</div><div>'
                    '<div class="series-label"><span class="series-dot scenario"></span>Scenario</div>'
                    f'<div class="compare-value">{row["after"]}</div>{after_bar}</div></div>'
                    f'<div class="change-callout">{row["sentence"]}</div></div>'
                )
            st.markdown(
                f"""
                <div class="comparison-card" role="figure" aria-label="{rows[0]['label']} in spring and fall">
                  <div class="comparison-title"><span>{rows[0]['label']}</span><span>Spring + Fall</span></div>
                  <div class="season-pair">{''.join(season_cells)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if not scenario.is_typical and effect_rows:
        validated_seasons = sorted({
            row["season"].replace(" migration", "").title()
            for row in effect_rows if row["support"] == "validated"
        })
        projected_seasons = sorted({
            row["season"].replace(" migration", "").title()
            for row in effect_rows if row["support"] != "validated"
        })
        evidence_detail = (
            f"{natural_join(validated_seasons)} beat both historical baselines on hidden years. "
            f"{natural_join(projected_seasons)} applies that tested response to its own recorded journeys as a paired estimate."
            if projected_seasons
            else "Both seasons beat two simple historical guesses in years hidden from training."
        )
        st.markdown(
            f"""
            <div class="evidence-check">
              <span class="check-icon">✓</span>
              <div><b>Evidence-tested seasonal pair</b>
              <span>{evidence_detail} It is a projection—not proof of cause.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        outcome_words = natural_join(sorted({row["label"].lower() for row in effect_rows}))
        investigator_question = (
            f"Can you spot journeys that match the predicted change in {outcome_words}—and journeys that do not?"
        )
    else:
        investigator_question = (
            "What stays similar across the years? What changes from one animal to another?"
        )
    st.markdown(
        f"""
        <div class="think-card">
          <b>Think like a scientist</b>
          <span>{investigator_question}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

focus_seasons = []
route_geometry = cached_route_geometry(result, tuple(focus_seasons), not scenario.is_typical)
progress_key = f"progress_{selected_species}_{selected_preset.key}"
playing_key = f"playing_{selected_species}_{selected_preset.key}"
st.session_state.setdefault(progress_key, 0)
st.session_state.setdefault(playing_key, False)
map_rule = (
    "Purple = spring scenario · yellow = fall scenario"
    if has_route_effect
    else "Purple and yellow show evidence-weighted journeys; result cards identify the tested outcome"
    if has_stopover_effect
    else "Thin lines are recorded journeys · thicker lines summarize the population"
)


@st.fragment(run_every=0.18 if st.session_state[playing_key] else None)
def migration_player():
    is_playing = bool(st.session_state[playing_key])
    initial_progress = int(st.session_state[progress_key])
    button_label = (
        "⏸ Pause"
        if is_playing
        else "↻ Replay journey" if initial_progress >= 100 else "▶ Play journey"
    )
    play_column, tracker_column = st.columns(
        [0.9, 2.1], gap="medium", vertical_alignment="center"
    )
    with play_column:
        toggle_requested = st.button(
            button_label,
            type="secondary" if is_playing else "primary",
            use_container_width=True,
            key=f"play_button_{selected_species}_{selected_preset.key}",
        )
    if toggle_requested:
        if is_playing:
            is_playing = False
        else:
            if initial_progress >= 100:
                st.session_state[progress_key] = 0
            is_playing = True
        st.session_state[playing_key] = is_playing

    reached_arrival = False
    if is_playing:
        st.session_state[progress_key] = advance_migration_progress(
            int(st.session_state[progress_key]), step=1
        )
        if int(st.session_state[progress_key]) >= 100:
            is_playing = False
            reached_arrival = True
            st.session_state[playing_key] = False

    with tracker_column:
        progress_percent = st.slider(
            "Journey progress",
            min_value=0,
            max_value=100,
            step=1,
            key=progress_key,
            format="%d%%",
            disabled=is_playing,
            help="Pause the journey to move this tracker yourself.",
        )
        st.markdown(
            '<div class="timeline-labels"><span>Departure</span><span>Halfway</span><span>Arrival</span></div>',
            unsafe_allow_html=True,
        )
    spring_key = '<span style="color:#4191cd">● Spring baseline</span>'
    fall_key = '<span style="color:#2da578">● Fall baseline</span>'
    if effect_rows:
        experiment_key = (
            '<span style="color:#a45cff">● Spring scenario</span>'
            '<span style="color:#ffd54a">● Fall scenario</span>'
        )
        season_keys = f"{spring_key}{fall_key}"
    else:
        experiment_key = ""
        season_keys = f"{spring_key}{fall_key}"
    st.markdown(
        f"""
        <div class="quest-key">
          {season_keys}
          {experiment_key}
          <span>● Route travelers are {progress_percent}% through the journey</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pydeck_chart(
        render_population_deck(
            route_geometry,
            progress=progress_percent / 100.0,
            playback_mode=is_playing,
        ),
        use_container_width=True,
        height=480,
        key=f"migration_map_{selected_species}_{selected_preset.key}",
    )
    if is_playing:
        status = "Playing smoothly · recorded tracks return when paused"
    elif progress_percent >= 100:
        status = "Arrived · press Replay to start again"
    else:
        status = "Paused · drag the timeline or press Play"
    st.markdown(
        f"""
        <div class="map-status">
          <span><b>{status}</b></span>
          <span>{map_rule}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if toggle_requested or reached_arrival:
        # Render every keyed widget once before the full rerun so Streamlit keeps
        # the exact progress while changing or stopping the fragment timer.
        st.rerun()


map_explanation = (
    "Four travelers compare the full seasonal pair. Scenario routes share each known journey's departure and arrival anchors."
    if has_route_effect
    else "Four travelers compare evidence-weighted journeys; the result cards identify stopovers as the tested outcome."
    if has_stopover_effect
    else ""
)
map_explanation_html = f"<p>{map_explanation}</p>" if map_explanation else ""
with map_column:
    st.markdown(
        f"""
        <div class="map-heading">
          <div><div class="eyebrow">Watch the journey</div><h2>Migration map</h2>
          {map_explanation_html}</div>
          <span class="data-badge">{displayed_journeys} recorded journeys</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    migration_player()

with st.expander("🧑‍🏫 Teacher lab: evidence, methods, and sources"):
    if validations:
        compact_columns = [
            "season", "outcome", "years", "folds", "seasonal_error",
            "persistence_error", "environmental_error", "skill_vs_seasonal",
            "skill_vs_persistence", "fold_win_rate", "outcome_use",
        ]
        evidence = pd.DataFrame(validations)
        evidence["outcome_use"] = evidence.status.map(
            {
                "active historical association": "Scenario effect",
                "not activated": "Model context",
            }
        ).fillna("Model context")
        st.dataframe(
            evidence[[column for column in compact_columns if column in evidence]],
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("Historical weather has not been refreshed for this population yet.")
    if source_trials:
        st.markdown("**Environmental sources included in the model**")
        st.caption(
            "Every available source below is included. The evidence result separately "
            "shows whether that source improved predictions by itself on unseen years."
        )
        source_evidence = pd.DataFrame(source_trials)
        source_evidence["model_status"] = "Included"
        independent_status = source_evidence.get(
            "evidence_status", source_evidence.get("status", "")
        )
        source_evidence["independent_evidence"] = pd.Series(
            independent_status, index=source_evidence.index
        ).map(
            {
                "active historical association": "Passed both baselines",
                "not activated": "Did not beat both alone",
            }
        ).fillna("Diagnostic only")
        source_columns = [
            "season", "outcome", "source_group", "model_status",
            "independent_evidence", "folds", "skill_vs_seasonal",
            "skill_vs_persistence", "fold_win_rate",
        ]
        st.dataframe(
            source_evidence[
                [column for column in source_columns if column in source_evidence]
            ],
            hide_index=True,
            width="stretch",
        )
    st.markdown(
        """
        **Evidence rule:** All sufficiently covered environmental sources are included in the
        fitted model. An outcome may drive a classroom scenario only when the combined model
        beats both the prior-years seasonal mean and the
        previous-year persistence baseline on unseen future years. It also needs at least three
        forward tests, more than 5% skill against both, wins in at least 60% of folds, and stable
        effect direction.

        **Map rule:** Thin lines remain recorded telemetry. The thicker baseline route is a
        progress-wise population center, and the animated traveler is interpolated smoothly
        along it. Purple spring and yellow fall scenario travelers appear during experiments.
        Corridor-choice routes begin and end at the corresponding recorded seasonal anchors;
        the receiving season uses a clearly labeled paired projection when only its reciprocal
        season passed the independent backtest.
        Stopover-only experiments draw evidence-weighted companion routes but do not
        claim that corridor choice was validated; their tested counts remain in the paired cards.
        These are modeled projections, not observed animal paths or new evidence. A student
        experiment is shown only when it changes a retained feature and produces a non-trivial
        activated response. Historical associations are not proof of causation.
        """
    )
    source_rows = telemetry[["source_doi", "license"]].drop_duplicates()
    st.markdown("**Tracking-data sources**")
    for source in source_rows.itertuples(index=False):
        st.markdown(f"- DOI `{source.source_doi}` · {source.license}")
    st.caption(
        f"Classroom catalog: 3 featured animals selected from {len(manifest)} openly licensed studies. "
        "Long-distance thresholds: "
        "300 km aerial, 200 km marine, and 50 km terrestrial; at least three spring and three fall journeys."
    )
