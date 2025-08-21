
# Kaggle Clean Starter

End-to-end scaffold with reproducible steps:

1) `python scripts/make_processed.py` — raw -> processed features
2) `python scripts/train.py`         — training + CV + OOF + artifacts
3) `python scripts/evaluate.py`      — holdout/OOF report images + metrics json
4) `python scripts/predict.py`       — test inference -> submission.csv

## Data expectation
Place your raw files under `data/raw/`:

- `train.csv` containing the target label (default: `Survived` -> mapped to `target`)
- `test.csv` without the target

Adapt `scripts/make_processed.py` and `src/features/build.py` to your schema.
