from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import make_scorer
import numpy as np
from src.ml.shuffle import shuffle
import pandas as pd
import dask_ml.model_selection as DaskCV
from dask.diagnostics import ProgressBar
import warnings
from math import floor

def gridCV(model, param_grid:Dict, X:pd.DataFrame, Y:pd.Series, grid_cv_split = 5, verbose=False,
           is_regression=True):
    
    ''' A wrapper around sckit learn Grid CV implementation
    To manage what gets exposed...'''

    if (floor(X.shape[0]/grid_cv_split) < 2):
        grid_cv_split = floor(X.shape[0]/2)

    ## ToDo: N repeats for Grid search optimisation
    if is_regression:
        scoring = 'r2'
    else:
        scoring = 'accuracy'

    gs = DaskCV.GridSearchCV(
        estimator= model,
        param_grid= param_grid,
        cv = grid_cv_split,
        n_jobs=10,
        scoring=scoring,
    )
    X, Y = shuffle(X, Y)

    if verbose:
        with ProgressBar():
            gs_fit_model = gs.fit(X, Y)
    else:
        gs_fit_model = gs.fit(X, Y)

    return gs_fit_model