
import pandas as pd
from src.paths import RAW, PROCESSED, MODELS
from src.features.build import build_features
from src.modeling.inference import load_bundle, predict_scores

def main():
    bundle = load_bundle(MODELS / "model_v1.joblib")
    test = pd.read_csv(RAW / "test.csv")
    feats = build_features(test)
    # keep only seen columns
    cols = bundle["num_cols"] + bundle["cat_cols"]
    for c in cols:
        if c not in feats.columns: feats[c] = 0
    scores = predict_scores(bundle, feats[cols])
    sub = feats.assign(pred=scores)[["pred"]]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sub.to_csv(PROCESSED / "submission.csv", index=False)
    print("Saved:", PROCESSED / "submission.csv")

if __name__ == "__main__":
    main()
