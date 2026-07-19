# WildVector

*A project for OpenAI's GPT-5.6 hackathon.*

WildVector is a classroom migration-weather lab covering three deeply modeled animals: turkey vultures (Eastern North American flyway), Arctic foxes (Canadian High Arctic), and blue whales (Northeast Pacific). Students compare recorded spring and fall journeys against model "what-if" weather scenarios, rendered as a live-animated map. Recorded journeys are never moved; scenario routes are labeled model projections, not observations.

## Quick start

Use Python 3.11–3.13. Python 3.14 is not supported by the pinned scientific stack.

Run `quickstart.bat` (double-click it, or run it from a terminal) to set up the virtual environment, install dependencies, build the local catalog and outcome models for the three built-in animals, and launch the app. It skips any step that's already done, so it's safe to re-run.

To do it by hand instead:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Builds the local catalog and trains the weather-scenario models for the
# three built-in animals. Without these, scenario experiments (anything
# beyond the baseline view) will refuse to render.
.\.venv\Scripts\python.exe scripts\refresh_catalog.py --continue-on-error
.\.venv\Scripts\python.exe scripts\refresh_environment.py
.\.venv\Scripts\python.exe scripts\train_population_models.py
.\.venv\Scripts\python.exe scripts\precompute_corridors.py

.\.venv\Scripts\python.exe -m streamlit run app.py
```

No application API key is required. Moving a student control never downloads data.

## Built with AI

WildVector was built end-to-end with [Codex](https://openai.com/codex), OpenAI's coding agent, running on GPT-5.6, as the sole development tool — no other AI coding assistant touched this repository. Codex's reasoning effort was switched deliberately by phase of work, not left on one setting throughout:

- **Light effort** produced the initial blueprint and implementation plan, and later handled polish passes once the app's shape was settled — naming, docstrings, README and data-responsibility language, UI copy.
- **High effort** was reserved for troubleshooting: getting the land/water A* routing correct, designing the outcome-model activation gate (the rolling-origin backtest that decides whether a scenario is allowed to change what students see), diagnosing Streamlit caching behavior for real-time playback, and fixing scenario-vs-baseline color-consistency bugs in the migration player.
