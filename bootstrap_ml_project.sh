#!/usr/bin/env bash
set -e

# 0) 디렉토리
mkdir -p notebooks data/raw data/interim data/processed
mkdir -p models/artifacts models/reports
mkdir -p src/{data,features,modeling,evaluation,utils}
mkdir -p scripts tests

# 1) __init__.py
touch src/__init__.py src/data/__init__.py src/features/__init__.py \
      src/modeling/__init__.py src/evaluation/__init__.py src/utils/__init__.py

# 2) 공용 경로/설정
cat > src/paths.py <<'PY'
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models" / "artifacts"
REPORTS = ROOT / "models" / "reports"
PY

cat > src/config.py <<'PY'
from dataclasses import dataclass
from typing import Optional

@dataclass
class CVConfig:
    n_splits: int = 5
    kind: str = "stratified"  # "kfold" | "stratified" | "group"
    random_state: int = 42
    shuffle: bool = True

@dataclass
class SearchConfig:
    n_iter: int = 40
    scoring: str = "roc_auc"
    n_jobs: int = -1
    verbose: int = 2

@dataclass
class TrainConfig:
    use_sample_weight: bool = True
    class_weight: Optional[str] = "balanced"
    thresh_basis: str = "proba"
    thresh: Optional[float] = None  # None이면 튜닝

cv_cfg = CVConfig()
search_cfg = SearchConfig()
train_cfg = TrainConfig()
PY

# 3) 데이터 스플릿/피처/모델 파이프라인 등 최소 구현
cat > src/data/split.py <<'PY'
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
def get_splitter(kind: str, n_splits: int, shuffle: bool, random_state: int):
    if kind == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if kind == "group":
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
PY

cat > src/features/build.py <<'PY'
import pandas as pd
import numpy as np

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
    # 예시: fam_size, age, fare 컬럼이 있다면 파생
    if "fam_size" in out.columns:
        out["fam_cat"] = out["fam_size"].apply(fam_categorized)
    if "age" in out.columns:
        out["age_bin"] = out["age"].apply(age_bin)
    if "fare" in out.columns:
        out["fare_bin"] = out["fare"].apply(fare_bin)
    return out
PY

cat > src/modeling/pipelines.py <<'PY'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_pipeline(model, num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe
PY

cat > src/modeling/search.py <<'PY'
from sklearn.model_selection import RandomizedSearchCV
def randomized_search(pipe, param_dist, cv, scoring, n_iter, n_jobs, random_state, verbose):
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=verbose,
    )
    return rs
PY

cat > src/modeling/threshold.py <<'PY'
import numpy as np
from sklearn.metrics import f1_score

def pick_threshold(clf, X, y_true, basis="proba", grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 81)
    if basis=="proba" and hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X)[:,1]
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X)
    else:
        return 0.5
    best, best_t = -1, 0.5
    for t in grid:
        y_pred = (score>=t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1>best:
            best, best_t = f1, t
    return best_t
PY

cat > src/evaluation/error_eda.py <<'PY'
import pandas as pd
import numpy as np

def build_eval_df(clf, X_df, y_true, thresh=None, basis="proba"):
    if basis=="proba" and hasattr(clf,"predict_proba"):
        score = clf.predict_proba(X_df)[:,1]
    elif hasattr(clf,"decision_function"):
        score = clf.decision_function(X_df)
    else:
        score=None
    y_pred = clf.predict(X_df) if score is None else (score>=thresh).astype(int)
    out = X_df.copy()
    out['y_true'] = np.asarray(y_true)
    out['y_pred'] = y_pred
    if score is not None: out['score']=score
    return out

def error_table(df, group_cols, min_count=20):
    if isinstance(group_cols,str): group_cols=[group_cols]
    tmp = df.copy()
    tmp['err'] = (tmp['y_true']!=tmp['y_pred']).astype(int)
    g = tmp.groupby(group_cols).agg(n=('err','size'), err_rate=('err','mean'))
    g = g[g['n']>=min_count].sort_values('err_rate', ascending=False)
    return g.reset_index()
PY

# 4) 최소 실행 스크립트(가공/학습/예측)
cat > scripts/make_processed.py <<'PY'
import pandas as pd
from src.paths import RAW, PROCESSED
from src.features.build import build_features

def main():
    # 예시: Titanic 형태 가정: raw/train.csv에 target 컬럼 'Survived' 존재
    df = pd.read_csv(RAW / "train.csv")
    # 예: fam_size가 없다면 만들어보기 (SibSp+Parch+1)
    if {"SibSp","Parch"} <= set(df.columns) and "fam_size" not in df.columns:
        df["fam_size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    if "Fare" in df.columns and "fare" not in df.columns:
        df["fare"] = df["Fare"]
    if "Age" in df.columns and "age" not in df.columns:
        df["age"] = df["Age"]
    if "Survived" in df.columns and "target" not in df.columns:
        df["target"] = df["Survived"]

    feats = build_features(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_csv(PROCESSED / "train_features.csv", index=False)
    print("Saved:", PROCESSED / "train_features.csv")

if __name__ == "__main__":
    main()
PY

cat > scripts/train.py <<'PY'
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.paths import MODELS, PROCESSED
from src.config import cv_cfg, search_cfg, train_cfg
from src.data.split import get_splitter
from src.modeling.pipelines import build_pipeline
from src.modeling.search import randomized_search
from src.modeling.threshold import pick_threshold

def main():
    df = pd.read_csv(PROCESSED / "train_features.csv")
    y = df["target"].values
    X = df.drop(columns=["target"])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = build_pipeline(LogisticRegression(max_iter=1000), num_cols, cat_cols)
    param_dist = {
        "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "model__penalty": ["l2"],
    }
    cv = get_splitter(cv_cfg.kind, cv_cfg.n_splits, cv_cfg.shuffle, cv_cfg.random_state)

    rs = randomized_search(pipe, param_dist, cv, search_cfg.scoring, search_cfg.n_iter,
                           search_cfg.n_jobs, cv_cfg.random_state, search_cfg.verbose)

    sw = compute_sample_weight(class_weight=train_cfg.class_weight, y=y) if train_cfg.use_sample_weight else None
    rs.fit(X, y, model__sample_weight=sw)

    t = train_cfg.thresh or pick_threshold(rs.best_estimator_, X, y, basis=train_cfg.thresh_basis)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rs.best_estimator_, "threshold": t,
                 "num_cols": num_cols, "cat_cols": cat_cols}, MODELS / "model_v1.joblib")
    print("Saved:", MODELS / "model_v1.joblib", "threshold:", t)

if __name__ == "__main__":
    main()
PY

cat > scripts/predict.py <<'PY'
import sys
import joblib
import pandas as pd
from src.paths import RAW, PROCESSED, MODELS

def main():
    bundle = joblib.load(MODELS / "model_v1.joblib")
    model = bundle["model"]
    # 예시: raw/test.csv → processed/test_features.csv
    test = pd.read_csv(RAW / "test.csv")
    if {"SibSp","Parch"} <= set(test.columns) and "fam_size" not in test.columns:
        test["fam_size"] = test["SibSp"].fillna(0) + test["Parch"].fillna(0) + 1
    if "Fare" in test.columns and "fare" not in test.columns:
        test["fare"] = test["Fare"]
    if "Age" in test.columns and "age" not in test.columns:
        test["age"] = test["Age"]

    # 여기선 간단히 원천 컬럼으로 바로 예측(실전은 build_features 재사용 권장)
    X = test.copy()
    num_cols = [c for c in bundle.get("num_cols", []) if c in X.columns]
    cat_cols = [c for c in bundle.get("cat_cols", []) if c in X.columns]
    # 누락된 컬럼 처리(간단 예시)
    for c in num_cols+cat_cols:
        if c not in X.columns: X[c] = 0

    preds = model.predict_proba(X[num_cols+cat_cols])[:,1]
    sub = pd.DataFrame({"id": test.index, "pred": preds})
    sub.to_csv(PROCESSED / "submission.csv", index=False)
    print("Saved:", PROCESSED / "submission.csv")

if __name__ == "__main__":
    main()
PY

# 5) pyproject (개발 설치)
if [ ! -f "pyproject.toml" ]; then
cat > pyproject.toml <<'TOML'
[project]
name = "kaggle-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pandas",
  "scikit-learn",
  "matplotlib",
  "seaborn",
  "joblib",
]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
TOML
fi

echo "Bootstrap complete."