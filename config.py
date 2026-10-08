from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RAW_CSV = DATA_DIR / "raw_sales.csv"
DB_PATH = DATA_DIR / "nexusiq.duckdb"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CHURN_WINDOW_DAYS = 45  # "churned" = no order in the last 45 days of data
