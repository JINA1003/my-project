
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

def metrics_from_scores(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true))>1 else None,
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    return out
