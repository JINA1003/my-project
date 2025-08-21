
import json
from pathlib import Path
import pandas as pd
import joblib

# Calculate paths directly
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models" / "artifacts"
REPORTS = ROOT / "models" / "reports"

def load_raw(name: str) -> pd.DataFrame:
    path = RAW / name
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, name: str):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / name, index=False)

def load_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)

def save_artifact(obj, name: str):
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, MODELS / name)

def load_artifact(name: str):
    return joblib.load(MODELS / name)

def save_report_json(obj, name: str):
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
