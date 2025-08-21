from setuptools import setup, find_packages

setup(
    name="kaggle-project",
    version="0.1.0",
    description="Reproducible Kaggle pipeline (data -> features -> modeling -> reports).",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
    include_package_data=True,
)
