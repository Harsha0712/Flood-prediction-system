"""Project settings. Keys come from environment variables or backend/.env only."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parent
load_dotenv(BACKEND / ".env")

BASIN_CODE = "003"  # Subernarekha. Change this to use another CWC basin file set.
START_YEAR = 2019
LOOKBACK = 90
HORIZON = 30
EVAL_HORIZONS = (1, 7, 15, 30)

DATA_DIR = BACKEND / "data_files"
MODEL_DIR = BACKEND / "trained_models"
THRESHOLDS = BACKEND / "config" / "thresholds.csv"
RESULTS = BACKEND / "results.json"

WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") or None
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or None
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",")
