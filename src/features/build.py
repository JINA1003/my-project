
import numpy as np
import pandas as pd

def fam_categorized(size):
    if size == 1: return "S"
    if size >= 5: return "L"
    return "M"

def age_bin(age):
    if pd.isna(age): return np.nan
    if age <= 12: return "child"
    if age <= 18: return "teen"
    if age <= 60: return "adult"
    return "elderly"

def fare_bin(fare):
    if pd.isna(fare): return np.nan
    if fare < 10: return "low"
    if fare < 30: return "mid"
    return "high"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Titanic-style mapping convenience
    if {"SibSp","Parch"} <= set(out.columns) and "fam_size" not in out.columns:
        out["fam_size"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    if "Fare" in out.columns and "fare" not in out.columns:
        out["fare"] = out["Fare"]
    if "Age" in out.columns and "age" not in out.columns:
        out["age"] = out["Age"]

    # categorical bins
    if "fam_size" in out.columns:
        out["fam_cat"] = out["fam_size"].apply(fam_categorized)
    if "age" in out.columns:
        out["age_bin"] = out["age"].apply(age_bin)
    if "fare" in out.columns:
        out["fare_bin"] = out["fare"].apply(fare_bin)
    return out
