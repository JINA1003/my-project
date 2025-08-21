
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
    return float(best_t)
