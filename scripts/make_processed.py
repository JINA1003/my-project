
import pandas as pd
from src.paths import RAW, PROCESSED
from src.data.io import save_processed
from src.features.build import build_features

def main():
    df = pd.read_csv(RAW / "train.csv")
    # Map target to 'target' if typical Titanic
    if "survived" in df.columns and "target" not in df.columns:
        df["target"] = df["survived"]
    feats = build_features(df)
    save_processed(feats, "train_features.csv")
    print("Saved:", PROCESSED / "train_features.csv")

if __name__ == "__main__":
    main()
