
import numpy as np
import pandas as pd

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
