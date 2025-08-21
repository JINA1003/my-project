
from sklearn.model_selection import RandomizedSearchCV

def randomized_search(pipe, param_dist, cv, scoring, n_iter, n_jobs, random_state, verbose):
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=verbose,
    )
    return rs
