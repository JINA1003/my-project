
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

def get_splitter(kind: str, n_splits: int, shuffle: bool, random_state: int):
    if kind == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if kind == "group":
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
