<<<<<<< HEAD
# River Water Level Forecasting and Early Flood Warning

College mini project. It forecasts river levels and rainfall for the next 30 days using deep learning, for stations in one Indian river basin (Subernarekha). Another CWC basin can be used by changing `BASIN_CODE` in `backend/settings.py`.

**Warnings are AI-generated decision-support information, not official government emergency notifications.**

## Outcomes

1. River water level forecasting (LSTM and CNN-LSTM, best one chosen on validation error)
2. Early flood prediction (threshold-crossing dates)
3. Flood risk classification (Normal, Warning, Danger, Critical)
4. Rainfall impact analysis (GRU rainfall forecast, correlation and delay)

## Run it

Use Python 3.11.

```bash
cd backend
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your free API keys from [OpenWeather](https://openweathermap.org/api) and [WeatherAPI](https://www.weatherapi.com/). The app still works without them, using the GRU model only.

```bash
.venv/Scripts/python train.py
.venv/Scripts/python forecast.py
.venv/Scripts/python -m uvicorn main:app --port 8000
```

`train.py` downloads the data and trains the models. `forecast.py` backtests them and writes `results.json`.

Then, in another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500.

## Flood thresholds

Add official CWC values to `backend/config/thresholds.csv`, with their source. They are published in CWC flood bulletins. Stations without thresholds show **Unavailable**, never Normal.

## Deploy

1. Push the repository to GitHub, including `backend/results.json`.
2. In GitHub Settings, under Pages, set the source to GitHub Actions.
3. On Render, create a Blueprint from the repository, then set `OPENWEATHER_API_KEY`, `WEATHERAPI_KEY` and `ALLOWED_ORIGINS` (your Pages address).
4. Put your Render URL in `frontend/js/config.js` and push.

Never commit `.env`. Render only serves `results.json`, so re-run the steps above and push to update forecasts.

## Limits

- Forecasts are uncertain, and beyond about a week they mainly reflect seasonal patterns.
- API forecasts cover only a few days. Every rainfall day is labelled with its source.
- Rainfall for a station is the IMD grid average within about 50 km, not true catchment rainfall.
- Correlation does not prove causation.

## Data sources

- [CWC river water level](https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc)
- [IMD gridded rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
=======
# Flood-prediction-system
>>>>>>> a1b96b93a96b5597aae661c6d6463c59ef775257
