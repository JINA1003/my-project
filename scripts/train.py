
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold

from src.paths import MODELS, REPORTS, PROCESSED
from src.config import cv_cfg, search_cfg, train_cfg
from src.data.io import load_processed, save_artifact, save_report_json
from src.data.split import get_splitter
from src.modeling.pipelines import build_pipeline
from src.modeling.params import LOGREG_SPACE
from src.modeling.search import randomized_search
from src.modeling.threshold import pick_threshold
from src.modeling.metrics import metrics_from_scores

def main():
    df = load_processed("train_features.csv")
    y = df["target"].values
    X = df.drop(columns=["target"])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    base_model = LogisticRegression(max_iter=1000)
    pipe = build_pipeline(base_model, num_cols, cat_cols)

    cv = get_splitter(cv_cfg.kind, cv_cfg.n_splits, cv_cfg.shuffle, cv_cfg.random_state)
    rs = randomized_search(pipe, LOGREG_SPACE, cv, search_cfg.scoring, search_cfg.n_iter,
                           search_cfg.n_jobs, cv_cfg.random_state, search_cfg.verbose)

    sw = compute_sample_weight(class_weight=train_cfg.class_weight, y=y) if train_cfg.use_sample_weight else None
    rs.fit(X, y, model__sample_weight=sw)

    # OOF scores (simple way using best_estimator_ inside CV folds again)
    # For a more exact OOF, do manual CV loop. Here we do a quick pass:
    oof_scores = np.zeros(len(y), dtype=float)
    splits = list(StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=cv_cfg.random_state).split(X, y))
    for tr_idx, va_idx in splits:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        sw_tr = compute_sample_weight(class_weight=train_cfg.class_weight, y=y_tr) if train_cfg.use_sample_weight else None
        m = rs.best_estimator_
        m.fit(X_tr, y_tr, model__sample_weight=sw_tr)
        if hasattr(m, "predict_proba"):
            oof_scores[va_idx] = m.predict_proba(X_va)[:,1]
        elif hasattr(m, "decision_function"):
            oof_scores[va_idx] = m.decision_function(X_va)
        else:
            oof_scores[va_idx] = m.predict(X_va).astype(float)

    t = train_cfg.thresh or pick_threshold(rs.best_estimator_, X, y, basis=train_cfg.thresh_basis)
    metrics = metrics_from_scores(y, oof_scores, threshold=t)

    # Persist
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {"model": rs.best_estimator_, "threshold": float(t),
              "num_cols": num_cols, "cat_cols": cat_cols}
    save_artifact(bundle, "model_v1.joblib")
    save_report_json({"cv_best_params": rs.best_params_, "cv_best_score": rs.best_score_, "oof_metrics": metrics},
                     "train_report_v1.json")
    oof_df = pd.DataFrame({"y_true": y, "score": oof_scores})
    (REPORTS / "oof_v1.csv").parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(REPORTS / "oof_v1.csv", index=False)

    print("Saved artifacts at:", MODELS / "model_v1.joblib")
    print("OOF metrics:", metrics)

if __name__ == "__main__":
    main()
