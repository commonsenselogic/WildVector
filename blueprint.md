# WildVector population-migration blueprint

## Product goal

WildVector lets children explore how seasonal weather relates to full migration patterns. The analytical unit is **species × named population × biological season × year**, assembled from multiple animals and years. The student view places spring and fall in one map while keeping their journeys separate.

The product must distinguish:

1. recorded long-distance spring and fall journeys;
2. natural variation among those tracked journeys;
3. a clearly labeled model-generated population route under a classroom weather story.

It may present a synthetic population projection only when it is derived from evidence-weighted recorded journeys, kept separate by biological season, and labeled as modeled rather than observed. It must never present an unconstrained route as a tracked animal or as new evidence.

The Wild Vector student experience must make the model claim visible before the map: one changed condition, one or more measurable before/after outcomes, and one concise evidence limitation. It should feel like a short field expedition rather than an analytics dashboard. Product imagery and interface accents must follow the supplied navy, cyan, green, and yellow globe/animal artwork.

## Data architecture

```text
Movebank repository metadata and licensed CSVs
    → validate identity, license, taxa, and coordinates
    → normalized partitioned Parquet
    → local DuckDB catalog
    → long-distance spring/fall journey filter
    → species/population classroom view

species/population/season/year nodes
    → ERA5 atmosphere + optional GLORYS ocean history
    → departure, arrival, duration, pace, stopover, and corridor-choice outcomes
    → isolated seasonal outcome models
    → forward-held-out backtest against seasonal and persistence baselines
    → teacher diagnostic evidence only
```

Generated data belongs under `data/catalog/` and is reproducible rather than committed.

Featured-source branches extend the base atmospheric history without combining populations: Bylot lemmings, ERA5-Land snow/soil conditions, and NSIDC sea ice for Arctic foxes; GLORYS physics plus chlorophyll, primary productivity, bathymetry, and SST fronts for blue whales; and hourly flight conditions plus derived uplift for turkey vultures. Acoustic, eBird, OBIS, and Motus products are independent validation or population-level targets only.

## Study and catalog contract

`catalog/studies.json` contains 25–50 selected packages. A package remains eligible only while its repository identity, accepted taxa, sensor data, and CC0 1.0 or CC BY 4.0 license validate. The current manifest contains 30 packages: 12 aerial, 9 terrestrial, and 9 marine.

Normalized telemetry contains:

```text
study_id, study_key, species, species_key,
population, population_key, animal_id,
timestamp_utc, latitude, longitude,
sensor_type, movement_type, hemisphere,
season, year, source_doi, license
```

The 30-study manifest is an expansion archive. The initial classroom catalog is deliberately limited to one deeply modeled population per movement domain: Eastern North American turkey vultures (avian), Northeast Pacific blue whales (aquatic), and Canadian High Arctic Arctic foxes (terrestrial). The fox story must say "seasonal movement and long-distance dispersal" because the population is partially migratory. Default catalog, environmental, and training jobs operate on these three; explicit research flags may expand beyond them.

Featured populations require at least three qualifying spring and three qualifying fall journeys. Qualifying net displacement is 300 km aerial, 200 km marine, or 50 km terrestrial. The loader balances both seasons and prioritizes the longest journeys.

`catalog/taxonomy.json` supplies a child-friendly common name and animal group for every accepted scientific taxon. Scientific names remain visible as secondary information.

## Recorded-journey representation

Each animal-season-year journey is sorted, deduplicated, and resampled to normalized progress for comparison. Spring and fall are never concatenated, even for the same animal and year. The student map draws thin recorded journeys and thicker progress-wise baseline population routes. During an experiment, four travelers move together: blue spring baseline, green fall baseline, purple spring scenario, and yellow fall scenario. Spring and Fall results are grouped together for every modeled outcome. Each activated scenario receives an evidence-weighted companion route that starts and ends at the corresponding known seasonal population anchors; the result card separately names the outcome that passed validation. Blue-whale routes use recorded range terminals off Guatemala and Southern California. Marine centerlines are snapped to ocean cells and routed around land. Sparse recorded marine fixes remain fixed, but any straight connector that would cross land is split instead of displayed. Non-spatial outcomes stay in result cards and do not add illustrative map dots or imply that corridor choice was validated.

## Evidence-based route scenario

Scenario execution never alters source telemetry. It reweights recorded journeys and generates a new progress-wise population route from their weighted latitude and circular longitude centers.

- Route-relative wind and current settings reweight observed journey pace; optional directional settings score route-bearing alignment.
- Temperature reweights observed departure timing within each season.
- Pressure contributes only a small timing preference.
- Weights are normalized within spring and fall so one season cannot erase the other.
- Purple means spring scenario and yellow means fall scenario; neither color represents observed telemetry or confidence. Separate scenario lines are used only for corridor choice; stopover counts remain in the paired result cards.
- A response that independently passes the activation gate in one season may be extended to the reciprocal season only as a labeled paired-season projection. Continuous effects transfer in standardized units around the receiving season's recorded baseline; corridor probability-point changes transfer to its recorded route mix.
- If the receiving season has no recorded alternate corridor, use the validated reciprocal-season deformation, reverse it by migration progress, and taper it to zero at the receiving season's known endpoints. This is a model projection, not separate held-out evidence.

This prevents cumulative velocity drift and keeps the projection inside the empirical population corridor as much as the recorded sampling supports. The projection is still synthetic and must remain visibly labeled as such.

## Environmental history and backtesting

ERA5 daily atmospheric features retain source, individual journey, outcome window, location, population, season, and year. Sample 30-day and 7-day pre-departure windows, three route-stage windows, and a 14-day arrival window. Include temperature range, humidity, pressure, rain, snow, light, radiation, evapotranspiration, wind/gusts, and route-aligned wind; select sea cells for marine animals. Train one isolated bundle per species-population-season for departure date, arrival date, duration, pace, stopovers, and corridor choice. Rolling evaluation trains only on earlier years and predicts every individual journey in the next unseen year. Compare every outcome with both the earlier-years seasonal median and the immediately previous year's population median.

Store fold count, feature coverage, environmental error, both baseline errors, both skill scores, fold win rate, coefficient-sign stability, and activation status. Activation requires at least three forward folds, more than 5% skill against both baselines, a 60% fold win rate, and stable effect direction. Diagnostics remain in the teacher panel. Activated outcomes may reweight recorded journeys and generate a labeled population route from those weighted coordinates; they may not rewrite telemetry. Any reciprocal-season transfer must be labeled as a paired estimate and must retain the independently validated source season in its metadata.

## Source retention and independent validation

Include every varying predictor with at least 35% multi-year coverage in the combined model, with imputation fitted independently inside each rolling training fold. Run the forward-held-out evaluation for every predictor source independently as a diagnostic, not as an inclusion switch. A combined outcome that fails its gate remains context-only even when one source passes alone. Store the included-source list, independently supported-source list, coverage, and evidence result in the population-season bundle. The teacher panel must label every fitted source “Included” and report the independent evidence result separately; it must never equate inclusion with predictive proof.

External acoustic, eBird, OBIS, and Motus products must be normalized to species, population, season, year, target, and observation count. They cannot supply telemetry route coordinates or leak into a held-out journey. Preserve dataset-specific licenses and distinguish validation from training targets.

## Student flow

1. arrive on a useful default migration with no setup click;
2. optionally choose a child-friendly animal-and-population label from one selector;
3. choose one animal-specific, evidence-gated what-if question;
4. see choices on the left, the migration map in the center, and the immediate baseline-to-scenario prediction on the right;
5. press Play to follow population travelers through 101 distance-spaced positions—one new position for every 1% increment—pause and drag the migration-progress tracker, or Replay from the start;
6. compare natural variation, the baseline, and the scenario outcome using a corridor only when route choice is supported.

There is no load button and no student-facing build configuration. The student view must not lead with tables, model metrics, or abstract coefficients. Telemetry and scenario results are cached locally for repeat play. Technical settings, licenses, model diagnostics, and limitations live in one collapsed teacher section. Controls update without source downloads.

## Acceptance criteria

- The manifest contains 25–50 validated open-license packages.
- The default classroom and build surface contains exactly one avian, one aquatic, and one terrestrial population selected from that archive.
- Every manifest taxon has a child-friendly common name and group.
- Student populations have repeated long-distance journeys in both migration seasons.
- Thin lines remain recorded telemetry; thicker baseline and scenario lines are clearly labeled modeled population routes.
- A scenario route is generated only for an activated corridor-choice outcome; marine routes must pass the water-only connector check.
- Every scenario shows a plain-language prediction and baseline-to-scenario outcome values before the map.
- The migration tracker interpolates a small number of population travelers continuously along the displayed baseline and scenario routes; recorded telemetry remains in the static thin paths.
- The student surface says when a result passed unseen-year tests and also says that this is not proof of cause.
- Every promised non-typical classroom run must change a retained model variable and produce a non-trivial activated response; the default training job fails if any preset loses support.
- Backtests predict future years and compare against both seasonal and persistence baselines.
- Unsupported historical models remain diagnostic and cannot activate scenario-route drawing.
- Cold app startup performs no telemetry or environmental network requests.
- Automated tests cover licensing, taxonomy, normalization, filtering, seasonal isolation, source-coordinate invariance, projected-route labeling, and validation.
