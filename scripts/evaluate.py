
import pandas as pd
from src.paths import REPORTS
from src.evaluation.report import save_curves

def main():
    oof_df = pd.read_csv(REPORTS / "oof_v1.csv")
    save_curves(oof_df, name_prefix="v1")
    print("Saved ROC curve under:", REPORTS)

if __name__ == "__main__":
    main()
