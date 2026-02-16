from sklearn.model_selection import RepeatedKFold, cross_validate
import pandas as pd
from src.ml.cross_validation_executor import CrossValidationExecutor
from math import floor

class KfoldCrossValidationExecutor(CrossValidationExecutor):

    def __init__(self):
        super().__init__()

    def execute(self, X : pd.DataFrame, Y : pd.DataFrame, model, standardisation=False, standardise_y=False, k_splits=5, n_repeats=2, verbose=False,
                is_regression = True):
        self.X = X
        self.Y = Y
        self.model = model
        if (floor(X.shape[0]/k_splits) < 2):
            k_splits = floor(X.shape[0]/2)
        cv = RepeatedKFold(n_splits=k_splits, n_repeats=n_repeats, random_state=123)
        cross_val_score = cross_validate
        return super().execute(X, Y, model, standardisation, standardise_y,
                            cv=cv, cross_val_score=cross_val_score, verbose=verbose,
                            is_regression=is_regression)