# Multi-River Water Level Forecasting and Early Flood Warning: Design

Date: 2026-11-07
Status: awaiting user review

## 1. Purpose

Build a working website that forecasts daily river levels and rainfall for the next 30 days across every eligible Indian monitoring station in the source datasets. It delivers exactly four outcomes:

1. River water level forecasting
2. Early flood prediction
3. Flood risk classification
4. Rainfall impact analysis

All alerts are AI decision support, not official government notifications.

Success means:

- The whole pipeline runs from a fresh checkout on real data, with no fabricated data or invented thresholds.
- Each published forecast shows its source, its uncertainty and its measured backtest error.
- Coverage gaps are reported rather than hidden.
- The site keeps working when both weather APIs are down.

Out of scope: water quality, pollution, discharge forecasting, reservoir inflow forecasting, user accounts and sending notifications.

## 2. Verified data findings

Checked 2026-11-07 by reading samples and dataset pages. No full file has been downloaded yet.

| Source | Finding | Design consequence |
|---|---|---|
| CWC telemetry (NWDP) | Public CSVs per basin and period. Columns include Station, State, District, River, Basin, Latitude, Longitude, RL_of_zeroGauge, Data Acquisition Time, level in metres. | Stations are discovered automatically. |
| Basin coverage | CWC files use basin codes 003-015, 26 and 27. CWC manual files add 28, 031 and 998. No CWC file was found for Indus, Ganga or Brahmaputra. | State-agency files (coded 999) are added as the route to those basins. Whether they contain those rivers is unverified until download. |
| History length | File periods overstate the data. One station's "1991-2020" file starts in June 2019. | Eligibility is measured per station, not assumed. |
| Placeholder values | Early rows at one station are exactly 0 against a gauge datum of 123 m. | Implausible values are removed in cleaning. |
| File sizes | Godavari telemetry: about 80 MB and 186 MB per period. | Read CSVs in chunks. The total is expected to be several GB. |
| IMD 0.25 grid | 1901-2024 stated on the IMD page. Available through `imdlib` and the IMD form. No login is shown. Download is untested. | Primary rainfall source, by year. |
| NWDP rainfall CSVs | District and state daily aggregates with no coordinates. The sampled rows begin 2026-08-19. | Gap filling only, not a training history. |
| Thresholds | Not in any CSV. No bulk dataset found. CWC bulletins carry warning, danger and HFL values, but list a station only while it is above warning level. | See section 5. |
| OpenWeather | Supplied key returns 200 for current weather and the 5-day forecast. | Available from day one. |
| WeatherAPI.com | Supplied key returns 401. | Needs its own free key. The app runs without it. |
| CWC flood site API | Undocumented. Inspecting it was blocked by the permission system. | Not used. It can replace section 5 if documented access is obtained. |

## 3. Architecture

```
NWDP CSVs ─┐
IMD grid ──┼─> pipeline (local) ─> daily parquet ─> training ─> checkpoints (local only)
Bulletins ─┘                             │                          │
                                         └────> forecast run <──────┘
                                                     │
                                         backend/artifacts/*.json.gz
                                                     │ committed to the repo
                       Render: FastAPI reads artifacts, adds live weather
                                                     │
                                  GitHub Pages frontend (static)
```

Two backend modes, set by `MODEL_RUNTIME`:

- `artifacts` (default, Render): serves precomputed results and installs `requirements-serve.txt`, which excludes PyTorch.
- `local`: also loads checkpoints, and the forecast run uses this mode. Installs `requirements.txt`, which includes PyTorch.

Reason: Render's free tier has too little memory to run PyTorch reliably. Consequence: forecasts refresh only when the pipeline is re-run, so every page shows the forecast generation time and the latest observation date.

## 4. Data pipeline

Project lives in `multi-river-flood-prediction/`, using the folder layout from the request. Additions needed to make it work: `backend/pipeline/`, `backend/training/`, `backend/config/`, `backend/artifacts/`, `backend/tests/` and `backend/requirements-serve.txt`.

**Sources manifest.** `backend/config/sources.json` lists NWDP dataset page URLs. The downloader reads each page, collects its CSV links and downloads what is missing into `data_raw/` (gitignored). New datasets are added here with no code change. Default entries: CWC telemetry, CWC manual, and the Assam, Bihar, Jharkhand and Uttar Pradesh telemetry datasets.

**Column detection.** Columns are matched by normalised name and alias lists, so differing files load with the same code. A file with no recognisable station, time or level column is skipped and logged.

**Station identity.** Key is the normalised (agency, station, river, state). Unrelated stations are never merged. The same station in telemetry and manual files is combined only when the name, river and coordinates agree within 2 km, with telemetry preferred on overlapping hours.

**Cleaning order.**

1. Remove exact duplicates.
2. Drop zero and blank placeholders, and readings below the station's gauge datum.
3. Convert any gauge-height series to reduced level by adding `RL_of_zeroGauge`, detected per station, so levels and thresholds share one datum.
4. Flag spikes using a rolling median and median absolute deviation test. Flagged values are dropped, never replaced with invented values.
5. Resample to daily mean and daily max. Max is what risk classification uses, because floods peak within a day. Mean is an input feature.
6. Interpolate gaps of up to 3 days. Longer gaps stay missing, and windows containing too much missing data are skipped.

**Rainfall matching.** Each station gets the mean of IMD grid cells within 50 km. The data has no catchment boundaries, so this is a stated proxy for catchment rainfall and is labelled as such. Daily rain uses this priority, with provenance kept per row:

1. IMD final gridded data
2. `imdlib` realtime data
3. NWDP district daily CSV, matched by station district

Water-level eligibility and rainfall training use years from 1990 onward.

**Eligibility.** A station gets forecasts when it has all of:

- at least 365 valid days covering at least one monsoon, with 60 or more June-September days
- at least 60 valid days in the last 90 days before the forecast origin
- a last observation no more than 14 days before the origin

The rest appear in a coverage report with the reason, and show on the map as grey.

**Features.** Previous-day level, level change, daily rain, 3, 7 and 14-day accumulated rain, 7 and 14-day rolling level means, month and day-of-year sine and cosine, and a station embedding. All rolling features look backward only.

**Splits.** Chronological per station: 70% train, 15% validation, 15% test. Scalers are fitted on training data only. Windows are never shuffled across split boundaries, though training batches are shuffled within the training period.

## 5. Flood thresholds

`backend/config/thresholds.csv` holds station key, warning, danger, HFL, optional critical level, source, source date and verified-by. Nothing outside this file is ever used as a threshold.

**Bulletin extraction.** `backend/pipeline/extract_thresholds.py` reads public CWC bulletin PDFs listed in `backend/config/bulletin_urls.txt`, taken from CWC's public bulletin pages, at no more than one request every 2 seconds. It parses the station rows. Format differences between years are handled per format and unparsed rows are logged.

**Matching.** Only exact matches on normalised station, river and state are accepted automatically. Near matches go to `thresholds_review.csv` for a person to confirm. When bulletins disagree the newest value wins and the conflict is logged, since HFL values are known to have been revised.

**Expected coverage: limited.** CWC reports about 200 level forecast sites nationally. A site reaches the file only if it was above warning on a collected bulletin and also appears in the water-level data, so coverage will be below 200 and may be far below it. Most stations will therefore show Unavailable. The real count is published on the dashboard. Asking CWC directly (fmdte@nic.in) for the consolidated list is the most direct way to improve this.

## 6. Models

All are global PyTorch models trained across stations, because per-station models are impractical on the available 2 GB GPU. Inputs use a 90-day lookback and outputs are 30 daily values.

- **GRU rainfall:** inputs are rainfall features, seasonal terms and the station embedding. Output uses softplus so it is never negative. Trained with Huber loss on log1p rainfall.
- **LSTM water level:** multivariate history encoder plus a 30-day future rainfall channel.
- **CNN-LSTM water level:** 1-D convolutions, then an LSTM, with the same inputs.

Water-level models predict the 30-day change from the last observed level, scaled by each station's own training spread, for both daily mean and daily max.

**Future rainfall channel without leakage.** The GRU trains first, on the training period only. Water-level models then receive GRU forecasts issued at each window's origin, never observed future rain. At inference, days covered by a weather API replace the GRU value. API forecast history does not exist, so that improvement cannot be backtested, and the site says so.

**Selection.** LSTM versus CNN-LSTM is chosen on validation RMSE, overall and per basin. The chosen model is recorded in `model_performance.json`. Checkpoints go to `backend/trained_models/` and are gitignored.

**Rainfall sources in the 30-day forecast.** API forecasts are used for the days they cover, about 5 for OpenWeather. The GRU fills the rest. Each day carries a `source` field. Seasonal rainfall beyond roughly a week tends toward climatology, and the rainfall page states this.

**Uncertainty.** P10 and P90 bands come from validation residuals, per horizon and per basin. Confidence labels are high for days 1-7, medium for days 8-15 and low for days 16-30.

## 7. Evaluation

Run on the test period using rolling origins spaced monthly, with no future observations as inputs.

- Regression: MAE, RMSE and R² at 1, 7, 15 and 30 days, for all three models.
- Reference baselines: persistence and day-of-year climatology, so model skill can be judged.
- Classification: accuracy, precision, recall, F1 and confusion matrix at the same horizons, only for stations with thresholds. Sample counts are shown, because flood days are rare and accuracy alone would mislead.

## 8. Flood analysis

**Daily classification,** using predicted daily max:

| Category | Rule |
|---|---|
| Normal | Below warning |
| Warning | At or above warning, below danger |
| Danger | At or above danger, below critical |
| Critical | At or above verified HFL or configured critical level |
| Unavailable | Thresholds missing, so the level cannot be placed |

If only some thresholds are known, a level at or above a known threshold takes that category or higher. A level below the lowest known threshold is Unavailable unless warning is known. Missing thresholds are never classified as Normal.

A station's headline category is its worst forecast day. Its current category is shown separately, and both appear in the 30-day calendar.

**Early warning states:**

- `already_crossed`: the latest observed level is at or above the threshold.
- `predicted_crossing`: the P50 forecast crosses it. Gives the first crossing date and days remaining.
- `possible_crossing`: only the P90 forecast crosses. This is the "approaching" test, and it uses the forecast band instead of an arbitrary universal margin.

Supporting signals are a one-day rise above the station's own 99th percentile, IMD heavy rain (64.5 mm or more per day) in the matched rainfall, and accumulated rainfall.

**Alerts** contain every field the request lists, use its four message templates, include the reason in plain words and carry the confidence label and the disclaimer. They are sorted by severity, then by days remaining. River and state rollups list where attention is needed.

**Rainfall impact.** Per station: historical rainfall, 3, 7 and 14-day accumulations, Spearman correlation with level and level change, lag correlation for 0-14 days, level rise after heavy-rain days, and a correlation heatmap of accumulation window by lag. Results for monsoon-only and all-year periods are shown, with a note that correlation does not prove causation.

## 9. Weather APIs

`backend/api_services/` holds one service per provider plus `api_manager.py`. Keys are read from environment variables, loaded from `backend/.env` locally. `.env` is gitignored and no key reaches the frontend or a log line.

- **Merging:** one record per station and date with a `sources` list. When both providers cover a full date, rainfall is their mean. Partial days are excluded from daily totals.
- **Failure order:** other provider, then recent cache, then GRU forecast. Responses say which applied.
- **Limits:** 10-second timeout, one retry with backoff on 429 or 5xx, and a daily budget counter per provider set below the free-plan limit. Plan limits are confirmed from provider docs during implementation.
- **Cache:** SQLite, 30 minutes for current weather and 3 hours for forecasts.
- **Batch calls:** forecast runs group stations into 0.5 degree cells and call once per cell, then stop at the budget. Remaining cells use the GRU, recorded per station.

## 10. Backend

FastAPI with CORS limited to the Pages origin and localhost. Provides the 11 requested `GET /api/...` endpoints, plus `GET /api/health`. List endpoints accept `river`, `state`, `basin`, `station_id` and `days` where relevant. Errors use one JSON shape. Missing artifacts return a clear 503 and do not crash the app.

Artifacts: stations, forecasts, 365-day observed history, risk, alerts, rainfall impact, model performance, coverage report and run metadata, stored as gzipped JSON. Counts on the dashboard are computed from these.

## 11. Frontend

Static HTML, CSS and JavaScript with Bootstrap 5, Chart.js and Leaflet, on the eight pages from the requested layout. The home page contains the dashboard stats and the India map. `js/app.js` holds the backend URL and shared fetch code. Grey markers and the coverage count are visible on the home page. Loading, empty and backend-offline states exist on each page. API-sourced and GRU-sourced rainfall are visually distinct, and forecast charts show the P10-P90 band.

## 12. Deployment

- `deploy-frontend.yml` publishes `frontend/` to GitHub Pages on push.
- `test-backend.yml` runs pytest on Python 3.11 with the serving requirements.
- `render.yaml` describes the service.
- The README covers everything in section 18 of the request, including key setup, dataset download, training, the refresh loop and pushing updated artifacts.

Publishing needs the user's own GitHub and Render accounts. Raw data, checkpoints and `.env` are gitignored. Only small result artifacts are committed.

## 13. Testing and verification

- Unit tests with small hand-written fixtures in `backend/tests/`, used only by tests, covering column detection, cleaning, feature leakage, split order, classification including missing thresholds, crossing logic, API merging and fallback.
- API tests with FastAPI's test client against sample artifacts.
- First real run on one basin (Subernarekha) through to the site, then all basins.
- Browser check of all pages at phone and desktop widths, with the backend offline and online.

Completion is reported with real station counts, threshold coverage, and backtest scores against baselines.

## 14. Known limits

- Coverage depends on the source data. Ganga, Yamuna and Brahmaputra appear only if state files contain them, and the site will report if they do not.
- Most stations are expected to lack thresholds and so show Unavailable.
- Many stations may have only a few years of history, which limits skill on rare floods.
- 30-day river forecasts are inherently uncertain. Beyond about a week they mainly reflect seasonal patterns.
- Training time on the MX450 is unknown until measured on the first basin. If it is too slow, the plan falls back to CPU overnight runs or fewer stations per epoch.
