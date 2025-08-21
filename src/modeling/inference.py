
import joblib
import numpy as np
import pandas as pd

def load_bundle(path):
    return joblib.load(path)

def predict_scores(bundle, X: pd.DataFrame):
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)
