# Multi-River Flood Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-data 30-day river level and rainfall forecasting site with flood risk classification, early warnings and rainfall impact analysis for all eligible Indian stations.

**Architecture:** A local Python pipeline downloads and cleans data, trains shared PyTorch models and writes gzipped JSON artifacts. A FastAPI backend serves those artifacts and live weather. A static GitHub Pages site reads the backend.

**Tech Stack:** Python 3.11, PyTorch, Pandas, NumPy, Scikit-learn, Xarray, imdlib, pyarrow, pypdf, httpx, FastAPI, Uvicorn, pytest; HTML, CSS, JavaScript, Bootstrap 5, Chart.js, Leaflet.

**Spec:** `docs/superpowers/specs/2026-11-07-multi-river-flood-prediction-design.md`

## Global Constraints

- Python 3.11. No Streamlit, TensorFlow or Keras. Free API plans only.
- No fabricated data, fake predictions or invented thresholds. Hand-written fixtures live only in `backend/tests/`.
- Never put API keys in source, logs, frontend files or the README. `.env` is gitignored.
- Raw data and checkpoints are never committed. Only small artifacts are.
- Time-series order is preserved across split boundaries. Rolling features look backward only. Backtests use no future observations.
- Missing thresholds are never classified as Normal.
- Every alert is labelled as AI decision support, not an official government notification.
- Do not claim APIs provide 30 days of forecasts. Each rainfall day carries a `source`.
- Do not add water quality or pollution features.
- `LOOKBACK_DAYS = 90`, `HORIZON_DAYS = 30`. Evaluation horizons: 1, 7, 15, 30.
- All commands run from `backend/` with `.venv/Scripts/python` (Windows). Commit messages end with the attribution line from the session.

## Review Focus

- CWC timestamps are day-first: `13-05-2024 05:30` must parse as 13 May, and `01-02-2021` as 1 February, never January 2. Test in Task 2.
- A provider error must not expose the API key, because HTTP errors include the request URL with the key in it. Logs and responses must be key-free. Test in Task 3.
- A threshold row with warning above danger, or danger above HFL, must go to review and never drive a classification. Test in Task 10.
- `days=0`, `days=31` and `days=abc` must return 422, and an unknown `station_id` must return 404. A filter with no matches returns an empty list. Test in Task 14.
- Text filters must ignore case and spacing, because sources mix `UTTAR PRADESH` and `Uttar Pradesh`. Test in Task 14.

---

### Task 1: Project scaffold

**Files:**
- Create: `.gitignore`, `backend/requirements.txt`, `backend/requirements-serve.txt`, `backend/.env.example`, `backend/.env`, `backend/pytest.ini`, `backend/settings.py`, `backend/config/sources.json`, `backend/config/bulletin_urls.txt`
- Test: `backend/tests/test_settings.py`

**Interfaces:**
- Produces: `settings.get_settings() -> Settings` with `weatherapi_key: str | None`, `openweather_api_key: str | None`, `model_runtime: str` (default `"artifacts"`), `allowed_origins: list[str]`, and path fields `raw_dir`, `processed_dir`, `artifacts_dir`, `checkpoints_dir`; constants `LOOKBACK_DAYS`, `HORIZON_DAYS`, `EVAL_HORIZONS`.

- [ ] **Step 1:** Run `git init` in the project folder, create `.venv` with `py -3.11 -m venv backend/.venv`, and install CUDA-enabled `torch` plus the requirements, then confirm `torch.cuda.is_available()`. If it is False, continue on CPU and note it.
- [ ] **Step 2:** Write `.gitignore` covering `.env`, `.venv/`, `__pycache__/`, `data_raw/`, `data_processed/`, `backend/trained_models/*.pt`. Put the user's key in `backend/.env` as `OPENWEATHER_API_KEY`; leave `WEATHERAPI_KEY` blank. `.env.example` has only the placeholder values from the spec.
- [ ] **Step 3: Write failing tests:** `test_keys_read_from_env` (monkeypatched env values appear in settings), `test_missing_keys_are_none`, `test_constants` (`LOOKBACK_DAYS == 90`, `HORIZON_DAYS == 30`, `EVAL_HORIZONS == (1, 7, 15, 30)`).
- [ ] **Step 4:** Run `.venv/Scripts/python -m pytest tests/test_settings.py -v`. Expected: FAIL, module missing.
- [ ] **Step 5:** Implement `settings.py` with `python-dotenv`; `sources.json` lists the six water-level dataset page URLs from spec section 4 plus the NWDP daily rainfall dataset page. `pytest.ini` declares a `torch` marker for tests that need PyTorch.
- [ ] **Step 6:** Run the tests. Expected: PASS. Then run `git check-ignore backend/.env`. Expected: prints the path.
- [ ] **Step 7:** Commit everything except `.env`, including the existing spec and this plan.

### Task 2: Dataset loading

**Files:**
- Create: `backend/data/data_loader.py`, `backend/pipeline/download_data.py`
- Test: `backend/tests/test_data_loader.py`

**Interfaces:**
- Consumes: `get_settings()`.
- Produces: `extract_csv_links(html: str) -> list[str]`; `detect_columns(columns: list[str]) -> dict[str, str]`; `iter_water_csv(path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]`, yielding canonical columns `station, agency, state, district, river, basin, latitude, longitude, rl_zero, timestamp, level`; `load_rainfall_csv(path: Path) -> pd.DataFrame` with `state, district, date, rain_mm`.

- [ ] **Step 1: Write failing tests:** `test_extract_links_unique_csv_only`; `test_detect_columns_real_header` using the real CWC header, including `River Water Level Telemetry Hourly (meter)` mapping to `level`; `test_detect_columns_missing_required_raises`; `test_day_first_timestamp` asserting `13-05-2024 05:30` gives 2024-05-13 and `01-02-2021 05:30` gives 2021-02-01; `test_rainfall_header_with_newlines` covering `"Weekly \nActual"` style headers; `test_missing_optional_columns_become_na`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the loader with alias lists over lower-cased, whitespace-collapsed names. Parse time with `dayfirst=True`, explicit format first.
- [ ] **Step 4:** Implement `download_data.py`: reads `sources.json`, fetches each page, resumes by skipping files whose size matches `Content-Length`, waits 1 second between files, writes `data_raw/manifest.csv` with URL, size and time.
- [ ] **Step 5:** Run tests. Expected: PASS.
- [ ] **Step 6:** Download only the Subernarekha CWC files with `.venv/Scripts/python -m pipeline.download_data --match cwc_003` and load one through `iter_water_csv`. Expected: rows with non-null station, timestamp and level.
- [ ] **Step 7:** Commit.

### Task 3: Weather API integration

**Files:**
- Create: `backend/api_services/__init__.py`, `weatherapi_service.py`, `openweather_service.py`, `api_manager.py`, `api_cache.py`
- Test: `backend/tests/test_api_services.py`

**Interfaces:**
- Produces: each service has `get_current(lat, lon) -> dict` and `get_forecast(lat, lon) -> list[DayRain]`, where `DayRain` has `date: date, rain_mm: float, complete: bool`, and raises `ProviderError(provider, status)`. `ApiCache(path).get(key)`, `.set(key, value, ttl_seconds)`, `.spend(provider, limit) -> bool`. `ApiManager(cache, services).current(lat, lon) -> dict` and `.forecast(lat, lon) -> {"days": [{"date", "rain_mm", "sources"}], "fallback": "none" | "other_provider" | "cache" | "model"}`.

Use `httpx.MockTransport` in tests, never real keys.

- [ ] **Step 1: Write failing tests:** `test_openweather_sums_3h_rain_into_days` marks partial first and last days as `complete=False`; `test_merge_one_record_per_date` with both providers giving 10 and 20 mm yields 15 and both sources; `test_partial_day_excluded`; `test_one_provider_down_uses_other` gives `fallback == "other_provider"`; `test_both_down_uses_cache`; `test_no_cache_returns_model_fallback` with empty days and no exception; `test_timeout_is_10_seconds`; `test_429_retries_once`; `test_budget_exhausted_skips_call`; `test_error_never_contains_key`, asserting `caplog` text and `str(ProviderError)` exclude a sentinel key on a 401 and on a connection error.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Pass keys through `params`, and catch `httpx.HTTPError` to re-raise `ProviderError` without the original message. Cache TTLs: 1800 seconds for current, 10800 for forecast. Set budgets from provider docs, checked in this task, below the free limit.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Run a live connectivity script, `.venv/Scripts/python -m api_services.api_manager --lat 17.0 --lon 81.8`. Expected: OpenWeather days with `sources` and WeatherAPI skipped without crashing.
- [ ] **Step 6:** Commit.

### Task 4: Water-level cleaning

**Files:**
- Create: `backend/preprocessing/data_preprocessing.py`, `backend/pipeline/build_water.py`
- Test: `backend/tests/test_cleaning.py`

**Interfaces:**
- Consumes: `iter_water_csv`.
- Produces: `station_id(agency, state, river, station) -> str`, a normalised slug; `clean_station(df) -> pd.DataFrame`; `to_daily(df) -> pd.DataFrame` with `station_id, date, level_mean, level_max, n_obs`; `merge_telemetry_manual(tele, manual) -> pd.DataFrame`. Pipeline output: `data_processed/water_daily.parquet`, `data_processed/stations.parquet`.

- [ ] **Step 1: Write failing tests:** `test_duplicates_removed`; `test_zero_placeholder_dropped` for 0 with datum 123; `test_below_datum_dropped`; `test_gauge_height_converted`, where median 2.3 with datum 123 becomes 125.3; `test_reduced_level_unchanged`; `test_spike_flagged_and_dropped`, with a genuine multi-day rise kept; `test_daily_max_and_mean`; `test_gap_of_3_filled_gap_of_4_not`; `test_different_stations_not_merged`; `test_merge_requires_within_2km`; `test_telemetry_wins_overlap`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Spike test: rolling 7-day median and MAD with threshold 8 MAD, and a value is dropped only if the next reading also returns near the median.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Run `build_water` on the Subernarekha files and print station count, date range and share of rows dropped. Expected: a plausible count and fewer than half of rows dropped; investigate otherwise.
- [ ] **Step 6:** Commit.

### Task 5: Rainfall data

**Files:**
- Create: `backend/pipeline/build_rainfall.py`
- Modify: `backend/preprocessing/data_preprocessing.py`
- Test: `backend/tests/test_rainfall_matching.py`

**Interfaces:**
- Produces: `cells_within_km(lat, lon, grid_lats, grid_lons, km=50) -> list[tuple[int, int]]`; `station_rain_from_grid(ds: xr.Dataset, stations: pd.DataFrame) -> pd.DataFrame`; `combine_rain_sources(grid, realtime, district) -> pd.DataFrame` with `station_id, date, rain_mm, rain_source`. Output: `data_processed/rain_daily.parquet`.

- [ ] **Step 1: Write failing tests:** `test_cells_within_50km_excludes_far`; `test_nearest_cell_used_when_none_in_radius`; `test_grid_fill_values_ignored`, covering IMD's -999 fill; `test_mean_of_cells`; `test_priority_grid_over_realtime_over_district`; `test_district_match_ignores_case`; `test_provenance_column`; a NetCDF and `.grd`-derived dataset load the same way.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement, downloading IMD years 1990 onward through `imdlib`, cached in `data_raw/imd/`, and converting to xarray.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Run it for Subernarekha stations. Expected: no negative rain, and a monsoon peak in monthly means. If `imdlib` fails, stop and report rather than substituting another source silently.
- [ ] **Step 6:** Commit.

### Task 6: Features, eligibility and windows

**Files:**
- Create: `backend/preprocessing/features.py`, `backend/pipeline/build_dataset.py`
- Test: `backend/tests/test_features.py`

**Interfaces:**
- Consumes: `water_daily.parquet`, `rain_daily.parquet`.
- Produces: `add_features(df) -> pd.DataFrame`; `check_eligibility(df, origin) -> tuple[bool, str]`, reason codes `too_few_days`, `no_monsoon`, `sparse_recent`, `stale`; `chronological_split(df) -> dict[str, pd.DataFrame]` (70/15/15); `StationScaler.fit(train).transform(df)`; `WindowDataset(df, target, lookback=90, horizon=30)` yielding `{"past", "future", "station", "target", "last_level", "scale", "origin"}`. Output: `features.parquet`, `coverage.parquet`.

- [ ] **Step 1: Write failing tests:** `test_accumulated_rain_3_7_14`; `test_rolling_means`; `test_changing_future_does_not_change_past_features`, proving no leakage; `test_split_ordered_and_disjoint`; `test_scaler_uses_train_only`; `test_eligibility_each_reason`, with 365 valid days and 60 monsoon days as the boundaries, plus 60 of 90 recent and 14-day staleness; `test_windows_skip_excess_missing`; `test_window_shapes`; `test_new_station_needs_no_code_change`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Window targets are `(future_level - last_level) / scale` for `level_mean` and `level_max`, with `scale` the station's training spread.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 7: Model architectures

**Files:**
- Create: `backend/models/lstm_model.py`, `gru_model.py`, `cnn_lstm_model.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Produces: `LSTMForecaster`, `CNNLSTMForecaster`: `forward(past[B,90,F], future[B,30,G], station[B]) -> [B,30,2]`. `GRURainfall`: `forward(past, future, station) -> [B,30]`. All take `(n_features, n_future, n_stations, hidden=128, layers=2, dropout=0.2)`, plus `embed_dim=16`.

- [ ] **Step 1: Write failing tests:** output shapes for all three; `test_gru_non_negative` on random extreme input; `test_cnn_lstm_has_conv_and_lstm`; `test_backward_pass_gives_gradients`; `test_station_embedding_changes_output`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The GRU ends with `softplus`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 8: Training

**Files:**
- Create: `backend/training/train.py`, `backend/training/hindcast.py`
- Test: `backend/tests/test_training.py`

**Interfaces:**
- Consumes: `WindowDataset`, the models.
- Produces: `train_model(name: Literal["gru", "lstm", "cnn_lstm"], dataset_dir, epochs, batch_size) -> Path`; `gru_hindcast(checkpoint, features) -> pd.DataFrame` with `station_id, origin, lead, rain_mm`. Checkpoints include `state_dict`, station index, scalers and feature lists. CLI: `python -m training.train --model gru|lstm|cnn_lstm`.

- [ ] **Step 1: Write failing tests:** `test_overfits_small_batch`, where loss falls by at least half on a fixture; `test_checkpoint_round_trip`; `test_early_stopping_uses_validation`; `test_water_future_channel_is_hindcast_not_observed`, asserting changed observed future rain leaves training inputs unchanged; `test_gru_loss_on_log1p`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement Huber loss, Adam, gradient clipping, patience of 5, mixed precision on GPU, batch size 256 by default and 96 if the MX450 runs out of memory.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Train all three on Subernarekha, recording time per epoch and peak GPU memory in `backend/training/RUNLOG.md`. Expected: validation loss falls. This run answers the spec's training-time unknown.
- [ ] **Step 6:** Commit, excluding checkpoints.

### Task 9: Forecast generation

**Files:**
- Create: `backend/forecasting/rainfall_forecast.py`, `backend/forecasting/water_level_forecast.py`, `backend/forecasting/uncertainty.py`
- Test: `backend/tests/test_forecasting.py`

**Interfaces:**
- Consumes: checkpoints, `ApiManager.forecast`.
- Produces: `residual_bands(preds, actuals, basins) -> dict[str, list[dict]]`, keyed by basin with `{"lead", "p10", "p90"}` and a `"_all"` fallback; `confidence_label(lead: int) -> str`; `forecast_rain(station_row, features, model, api_days) -> list[{"date", "rain_mm", "source": "weatherapi"|"openweather"|"blended"|"gru"}]`; `forecast_level(station_row, features, rain_days, model, bands) -> list[{"date", "p50", "p10", "p90", "p50_max", "p90_max", "confidence", "lead"}]`; `plan_api_cells(stations, budget) -> dict[cell, list[station_id]]`.

- [ ] **Step 1: Write failing tests:** `test_30_days_always_returned`; `test_api_days_replace_gru_with_source_label`; `test_no_api_days_all_gru`; `test_rain_non_negative`; `test_future_channel_uses_api_values`; `test_level_is_last_plus_scaled_delta`; `test_p10_le_p50_le_p90`; `test_confidence_label_boundaries` for 7, 8, 15, 16; `test_basin_band_falls_back_to_all`; `test_cells_group_at_half_degree`; `test_budget_caps_cells`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Bands come from validation residuals only.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 10: Thresholds and risk classification

**Files:**
- Create: `backend/flood_analysis/risk_classification.py`, `backend/pipeline/extract_thresholds.py`, `backend/config/thresholds.csv`
- Test: `backend/tests/test_risk.py`, `backend/tests/test_thresholds.py`

**Interfaces:**
- Produces: `Thresholds(warning, danger, hfl, critical, source, source_date)`; `load_thresholds(path) -> tuple[dict[str, Thresholds], list[dict]]`, returning accepted rows and rejected rows; `classify(level: float, t: Thresholds | None) -> str`, returning `NORMAL|WARNING|DANGER|CRITICAL|UNAVAILABLE`; `risk_calendar(days, t) -> list[dict]`; `headline_category(calendar) -> str`; `parse_bulletin_text(text) -> list[dict]`; `match_threshold(row, stations) -> tuple[str | None, list[str]]`.

- [ ] **Step 1: Write failing classify tests** with thresholds 8, 10, 12:

```python
assert classify(7.99, t) == "NORMAL"
assert classify(8, t) == "WARNING"
assert classify(10, t) == "DANGER"
assert classify(12, t) == "CRITICAL"
assert classify(5, None) == "UNAVAILABLE"
# Partial thresholds
assert classify(11, T(danger=10)) == "DANGER"
assert classify(9, T(danger=10)) == "UNAVAILABLE"
assert classify(7, T(warning=8)) == "NORMAL"
assert classify(9, T(warning=8)) == "WARNING"
assert classify(11, T(hfl=12)) == "UNAVAILABLE"
assert classify(12, T(critical=12)) == "CRITICAL"
```

- [ ] **Step 2: Write failing threshold tests:** `test_warning_above_danger_rejected`; `test_danger_above_hfl_rejected`; `test_non_numeric_rejected`; `test_parse_2022_bulletin_row` and `test_parse_2024_bulletin_row` using text copied from the real samples; `test_exact_match_accepted`; `test_near_match_goes_to_review`; `test_newest_value_wins_and_logs_conflict`; `test_headline_is_worst_day`, with `UNAVAILABLE` never outranking a real category.
- [ ] **Step 3:** Run. Expected: FAIL.
- [ ] **Step 4:** Implement. Rejected rows go to `thresholds_review.csv`. Fill `bulletin_urls.txt` from CWC's public bulletin listing pages. Fetch at most one request every 2 seconds. Do not call the undocumented flood site API.
- [ ] **Step 5:** Run tests. Expected: PASS.
- [ ] **Step 6:** Run the extractor and print accepted, review and unparsed counts. Report the real coverage to the user.
- [ ] **Step 7:** Commit `thresholds.csv` with its source columns.

### Task 11: Early warnings

**Files:**
- Create: `backend/flood_analysis/early_warning.py`
- Test: `backend/tests/test_early_warning.py`

**Interfaces:**
- Consumes: `Thresholds`, `classify`, forecast day dicts.
- Produces: `find_crossings(latest_level, days, t) -> list[{"threshold", "state", "date", "days_remaining"}]`, `state` being `already_crossed|predicted_crossing|possible_crossing`; `rapid_rise(history) -> bool`; `build_alert(station, days, crossings, rain_days, t) -> dict | None`; `rollup(alerts) -> dict`; `sort_alerts(alerts) -> list`.

- [ ] **Step 1: Write failing tests:** `test_already_crossed_not_reported_as_predicted`; `test_first_crossing_date_and_days_remaining`; `test_only_p90_crossing_is_possible`; `test_no_threshold_no_crossings`; `test_rapid_rise_uses_station_99th_percentile`; `test_heavy_rain_at_64_5_mm`; `test_alert_has_all_required_fields`, listing river, state, station, date, predicted level, category, threshold, crossing date, reason, action, confidence, disclaimer; `test_templates_match_spec` with the four exact messages from the request; `test_sorted_by_severity_then_days`; `test_unavailable_station_gets_no_alert`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement, using `p90_max` for possible crossings and `p50_max` for predicted crossings.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 12: Rainfall impact analysis

**Files:**
- Create: `backend/flood_analysis/rainfall_impact.py`
- Test: `backend/tests/test_rainfall_impact.py`

**Interfaces:**
- Produces: `analyse_station(df) -> dict` with `monthly_rain`, `accumulations`, `spearman`, `lag_correlation` (lags 0-14, for monsoon and all-year), `heatmap` (windows 1, 3, 7, 14 by lag), `heavy_rain_response`, `n_days`, `note`.

- [ ] **Step 1: Write failing tests:** `test_lag_detected`, with a fixture where level follows rain by 3 days giving peak lag 3; `test_constant_series_returns_null_not_nan`; `test_too_few_points_returns_null`; `test_heavy_rain_event_response`; `test_output_is_json_safe`; `test_note_says_correlation_not_causation`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with `pandas` Spearman on daily level change, excluding gaps.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

### Task 13: Evaluation and artifact writer

**Files:**
- Create: `backend/evaluation/model_evaluation.py`, `backend/pipeline/write_artifacts.py`, `backend/pipeline/run_all.py`
- Test: `backend/tests/test_evaluation.py`, `backend/tests/test_artifacts.py`

**Interfaces:**
- Produces: `regression_metrics(y, p) -> {"mae", "rmse", "r2", "n"}`; `classification_metrics(y, p) -> {"accuracy", "precision", "recall", "f1", "confusion", "labels", "support"}`; `rolling_origins(test_df, every_days=30) -> list[date]`; `baseline_persistence(...)`, `baseline_climatology(...)`; `select_model(results) -> {"overall", "by_basin"}`; `write_artifacts(results, out_dir, metadata)`. Files: `stations`, `forecasts`, `history`, `risk`, `alerts`, `rainfall_impact`, `model_performance`, `coverage`, `run_metadata`, all `.json.gz`.

- [ ] **Step 1: Write failing tests:** `test_metric_values` on known numbers; `test_origins_inside_test_period`; `test_inputs_end_at_origin`; `test_persistence_repeats_last_value`; `test_climatology_uses_train_only`; `test_metrics_exclude_stations_without_thresholds`; `test_selects_lower_validation_rmse`; `test_all_artifacts_written`; `test_metadata_has_generated_at_and_latest_observation`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. `run_all.py` runs build, train if requested, evaluate, forecast and write, with `--basin` to limit scope.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Run `python -m pipeline.run_all --basin Subernarekha --skip-train`. Expected: all artifact files exist and are under 25 MB in total for this basin, with scores for all models and baselines.
- [ ] **Step 6:** Commit code only. Artifacts are committed in Task 17.

### Task 14: FastAPI backend

**Files:**
- Create: `backend/main.py`, `backend/artifact_store.py`, `render.yaml`
- Test: `backend/tests/test_api.py`, with sample artifacts in `backend/tests/fixtures/artifacts/`

**Interfaces:**
- Consumes: artifact files, `ApiManager`.
- Produces: routes `GET /api/rivers`, `/stations`, `/dashboard`, `/forecast/water-level`, `/forecast/rainfall`, `/flood-risk`, `/early-warnings`, `/rainfall-impact`, `/model-performance`, `/weather/current`, `/weather/forecast`, `/health`. Query params `river, state, basin, station_id`, `days: int = Query(30, ge=1, le=30)`. Error shape `{"error": {"code", "message"}}`.

- [ ] **Step 1: Write failing tests:** each route returns 200 with sample artifacts; `test_filter_ignores_case_and_spacing`; `test_no_match_empty_list`; `test_unknown_station_404`; `test_days_validation`, with 0, 31 and `abc` all 422; `test_days_limits_rows`; `test_dashboard_counts_from_artifacts`; `test_missing_artifacts_503_not_crash`, while `/api/health` stays 200; `test_weather_down_returns_model_fallback`; `test_cors_allows_configured_origin_only`; `test_no_key_in_any_response`; `test_serving_works_without_torch`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Import torch only when `MODEL_RUNTIME=local`. Artifacts load once and reload when `run_metadata` changes.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Start the app against real Subernarekha artifacts and call `/api/dashboard`. Expected: real counts.
- [ ] **Step 6:** Commit.

### Task 15: Frontend

**Files:**
- Create: `frontend/index.html`, `rivers.html`, `forecast.html`, `rainfall.html`, `flood-risk.html`, `alerts.html`, `rainfall-impact.html`, `model-performance.html`, `css/style.css`, `js/app.js`, `dashboard.js`, `map.js`, `forecast.js`, `alerts.js`

**Interfaces:**
- Consumes: the Task 14 JSON routes.
- Produces: `js/app.js` exports `API_BASE` and `apiGet(path, params)`, showing a shared offline banner on failure.

The frontend has no test runner. Each page gets a browser check at Step 3.

- [ ] **Step 1:** Build the shared shell, `app.js`, and the home page with stat cards, the coverage count, the forecast generation time and the Leaflet map, with marker colours green, yellow, orange, red and grey and popups with the spec's fields.
- [ ] **Step 2:** Build the remaining seven pages: search and filter table; level chart with P10-P90 band, forecast table, peak and trend; rainfall chart with API and GRU days visually distinct; risk summary and 30-day calendar; severity-sorted alerts with the disclaimer; rainfall impact charts with a heatmap and the correlation note; model metrics with baselines and confusion matrices, including sample counts.
- [ ] **Step 3:** Verify every page in the browser at 375 px and desktop with the backend running. Expected: no console errors, no horizontal scroll, real data.
- [ ] **Step 4:** Stop the backend and reload each page. Expected: the offline banner, no blank crash.
- [ ] **Step 5:** Confirm the frontend has no key with `git grep -nE "appid|WEATHERAPI_KEY|OPENWEATHER_API_KEY" frontend`. Expected: no output.
- [ ] **Step 6:** Commit.

### Task 16: Deployment files and README

**Files:**
- Create: `.github/workflows/deploy-frontend.yml`, `.github/workflows/test-backend.yml`, `README.md`

**Interfaces:**
- Consumes: all prior commands, which the README quotes exactly.

- [ ] **Step 1:** Write `deploy-frontend.yml`, deploying `frontend/` on push to `main`, and `test-backend.yml`, running pytest on Python 3.11 with `requirements-serve.txt`. Skip torch-marked tests there with `-m "not torch"`.
- [ ] **Step 2:** Write the README covering all eleven items in request section 18, API key steps for both providers and Render, the threshold workflow, known limits, and the refresh loop.
- [ ] **Step 3:** Install from `requirements-serve.txt` into a clean venv and run `pytest -m "not torch"`. Expected: PASS without torch.
- [ ] **Step 4:** Commit.

### Task 17: Full run and verification

**Files:**
- Create: `backend/artifacts/*.json.gz`
- Modify: `backend/training/RUNLOG.md`

- [ ] **Step 1:** Run `python -m pipeline.download_data` in the background. Expected: all manifest rows present. Report failures and whether Ganga, Yamuna and Brahmaputra stations exist.
- [ ] **Step 2:** Build the full dataset, then print totals of stations, eligible stations and exclusion reasons.
- [ ] **Step 3:** Train all three models on all eligible stations, using the Task 8 timing to choose settings. Before starting, tell the user the expected duration.
- [ ] **Step 4:** Run the full pipeline. Expected: artifacts for all eligible stations, under 95 MB in total. If larger, trim history before committing.
- [ ] **Step 5:** Run the whole test suite. Expected: PASS.
- [ ] **Step 6:** Recheck all eight pages in the browser against the full artifacts, with one and both weather APIs disabled. Expected: forecasts stay available and fallback labels are correct.
- [ ] **Step 7:** Confirm `git ls-files` shows no `.env`, `.pt` or raw data, and scan history for the key with `git log -p --all | grep -c "$OPENWEATHER_API_KEY"`. Expected: 0.
- [ ] **Step 8:** Commit, then report station counts, threshold coverage and backtest scores against baselines. Pushing to GitHub and connecting Render are the user's steps, or need their explicit go-ahead.
