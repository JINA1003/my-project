
from dataclasses import dataclass
from typing import Optional

@dataclass
class CVConfig:
    n_splits: int = 5
    kind: str = "stratified"  # 'kfold' | 'stratified' | 'group'
    random_state: int = 42
    shuffle: bool = True

@dataclass
class SearchConfig:
    n_iter: int = 30
    scoring: str = "roc_auc"
    n_jobs: int = -1
    verbose: int = 2

@dataclass
class TrainConfig:
    use_sample_weight: bool = True
    class_weight: Optional[str] = "balanced"
    thresh_basis: str = "proba"
    thresh: Optional[float] = None

cv_cfg = CVConfig()
search_cfg = SearchConfig()
train_cfg = TrainConfig()
