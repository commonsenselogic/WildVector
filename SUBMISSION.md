# WildVector: Making Migration Data Legible to Students

## Inspiration
I was inspired by how LLMs can put large, messy scientific datasets directly into students' hands. Movebank alone holds millions of GPS fixes from tagged animals, but a CSV of lat/lon pairs means nothing to a classroom. The gap wasn't data availability; it was *translation*.

## What it does
WildVector is a Streamlit app where students explore real Movebank migration telemetry for turkey vultures, Arctic foxes, and blue whales, then run "what-if" weather experiments and watch a live-animated map compare the recorded baseline journey against a model-projected scenario route, colored consistently so it's always clear which line is "what happened" versus "what the model predicts."

## How I built it
I built this iteratively using GPT 5.6: a blueprint and implementation plan up front, then repeated cycles of prototyping capped off with a dedicated pass for performance and polish.

Under the Streamlit/pydeck UI:

- **Corridor construction.** Each animal's journey isbroken down into progress stations. At each station, the centerline is a **weighted median** latitude and a **weighted circular mean** longitude.
- **Reweighting, not fabrication.** A scenario never invents coordinates — it scores each recorded journey against a validated effect and reweights it by exponential tilting, then recomputes the same weighted-median corridor over the same animals.
- **A statistical activation gate.** Each outcome model is trained with rolling-origin cross-validation and only allowed to influence what students see if it beats seasonal-median and persistence baselines.

## Challenges I ran into
The two hardest problems weren't rendering a map — they were **trustworthiness** and **speed**.

*Trustworthiness:* it's trivially easy to build a slider that produces a smooth, convincing, fabricated route. I gated this at both build time and runtime — if a shipped experiment doesn't have a model that passed validation, the app itself refuses to render rather than show an unvalidated effect. That gate exposed a subtler bug: the outcome model still reports its supported effects even for a *typical* (zero-delta) scenario, so a baseline preset could carry scenario-colored markers it shouldn't. The fix was gating display purely on whether the selected scenario is actually non-typical, not on whether the model "has an opinion."

*Speed:* live playback was recomputing the entire routing pipeline on every animation tick. I split the expensive, per-session-constant geometry from the cheap per-frame marker interpolation, cached the former by content across reruns, and pre-warmed every species/experiment combination at launch.

## Accomplishments I'm proud of
The method I'm most proud of is the **reciprocal season transfer**. Telemetry is asymmetric — a species might have enough fall journeys to validate a model but too few spring ones. Instead of showing nothing, or silently reusing the fall model as if independently proven, `_transfer_route_deformation` borrows the *shape* of the validated season's response and projects it onto the other season's known route, tagged `support = "paired-season projection"` so the UI always distinguishes measured from inferred. It's a small piece of statistical honesty that matters in a tool meant to teach kids what evidence looks like.

## What I learned
A data tool for education fails differently than a normal dashboard: a *beautiful* wrong answer is worse than an ugly missing one. I learned to design the modeling layer around refusal and that in an interactive geospatial tool, correctness under real-time constraints is its own discipline: a routing pipeline that's fine once per page load is a real architecture problem at 5 frames per second.

## What's next for WildVector
More datasets and modeling — more species, more populations, outcome models beyond corridor choice and stopover count — and getting it in front of local teachers for feedback and integration into lesson plans.
