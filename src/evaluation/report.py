
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..paths import REPORTS

def save_curves(oof_df: pd.DataFrame, name_prefix: str):
    REPORTS.mkdir(parents=True, exist_ok=True)
    # ROC curve (simple)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(oof_df['y_true'], oof_df['score'])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.savefig(REPORTS / f"{name_prefix}_roc.png", bbox_inches="tight")
    plt.close()
