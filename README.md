# Kaggle Clean - Titanic ML Project

A clean, modular machine learning project structure for Kaggle competitions with reproducible steps.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd Titanic
```

2. **Create and activate virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the package in development mode**

```bash
pip install -e .
```

**⚠️ Important**: The `pip install -e .` step is crucial for the `src` module to be recognized!

### Usage

Run the scripts in order:

1. **Feature Engineering**

```bash
python scripts/make_processed.py  # raw -> processed features
```

2. **Model Training**

```bash
python scripts/train.py           # training + CV + OOF + artifacts
```

3. **Model Evaluation**

```bash
python scripts/evaluate.py        # holdout/OOF report images + metrics json
```

4. **Prediction**

```bash
python scripts/predict.py         # test inference -> submission.csv
```

## 📁 Project Structure

```
Titanic/
├── data/
│   ├── raw/          # Place your raw data files here
│   ├── interim/      # Intermediate data
│   └── processed/    # Processed features
├── models/
│   ├── artifacts/    # Trained models
│   └── reports/      # Training reports
├── notebooks/        # Jupyter notebooks for EDA
├── scripts/          # Main execution scripts
├── src/              # Source code package
│   ├── data/         # Data processing utilities
│   ├── features/     # Feature engineering
│   ├── modeling/     # ML modeling utilities
│   └── evaluation/   # Evaluation and reporting
└── submissions/      # Kaggle submission files
```

## 📊 Data Requirements

Place your raw data files under `data/raw/`:

- **`train.csv`**: Training data with target column (default: `Survived` → mapped to `target`)
- **`test.csv`**: Test data without target column

The current setup expects Titanic dataset columns. Adapt `scripts/make_processed.py` and `src/features/build.py` to your schema.

## 🔧 Customization

### Adding New Features

Edit `src/features/build.py` to add new feature engineering functions.

### Modifying Data Processing

Edit `src/data/io.py` for custom data loading/saving logic.

### Changing Models

Edit `src/modeling/` modules to use different algorithms or pipelines.

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'src'" error**

   - Solution: Make sure you ran `pip install -e .`
   - This registers the `src` directory as a Python package

2. **Import errors in scripts**

   - Solution: Check that all `__init__.py` files exist in subdirectories
   - Verify the package is installed: `pip list | grep kaggle-clean`

3. **Path-related errors**
   - Solution: Ensure data files are in the correct directories
   - Check `src/paths.py` for path configurations

### Development Setup

For developers who want to modify the code:

1. **Install in editable mode** (already done with `pip install -e .`)
2. **Code changes are automatically reflected** (no need to reinstall)
3. **Use the package structure**:
   ```python
   from src.paths import RAW, PROCESSED
   from src.data.io import load_raw, save_processed
   from src.features.build import build_features
   ```

## 📝 Dependencies

Core dependencies (see `requirements.txt`):

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `pip install -e .`
5. Submit a pull request

## 📄 License

This project is part of the SK Family AI Camp.

---

**Note**: This project structure follows best practices for ML projects and can be easily adapted to other Kaggle competitions or ML projects.
