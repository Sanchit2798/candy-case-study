from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import src.ml.shuffle as shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import accuracy_score

def custom_r2(Y_actual, Y_pred):
    return r2_score(Y_actual,Y_pred)

def custom_mape(Y_actual, Y_pred):
    '''We are multiplying scikit mape with 100 '''
    return mean_absolute_percentage_error(Y_actual, Y_pred) * 100

def custom_mae(Y_actual, Y_pred):
    return mean_absolute_error(Y_actual, Y_pred)

def custom_stdae(Y_actual, Y_pred):
    return np.std( Y_actual - Y_pred)

def custom_classification_accuracy(Y_actual, Y_pred):
    return accuracy_score(Y_actual, Y_pred) 


class CrossValidationExecutor:

    def __init__(self):
        pass

    def execute(self, X, Y, model, standardisation, standardise_y, cv, cross_val_score, verbose=False,
                is_regression= True):
        self.is_regression =  is_regression
        self.X = X
        self.Y = Y
        self.model = model
        self.cv = cv
        self.cross_val_score = cross_val_score
        if verbose:
            print('Cross Validation')
        X, Y = shuffle.shuffle(self.X, self.Y)

        if self.is_regression:
            scorers = {'R2 score' : make_scorer(custom_r2), 
                        'MAPE' : make_scorer(custom_mape),
                        'MAE' : make_scorer(custom_mae),
                        'STDAE' : make_scorer(custom_stdae)}
        else:
            scorers = {'Accuracy' : make_scorer(custom_classification_accuracy)}

        scorers, scores = self._get_fold_scores(standardisation, standardise_y, scorers, X, Y)
        cv_result_table = self._create_result_table(scores, scorers.keys(), verbose)
        return cv_result_table

    def _get_fold_scores(self, standardisation, standardise_y, scorers, X, Y):
        self.cv_obj = self.cv
        steps = list()
        
        if standardisation:
            steps.append(('scaler', StandardScaler()))
        
        if standardise_y and self.is_regression:
            steps.append(('model', TransformedTargetRegressor(regressor=self.model, transformer=StandardScaler())))
        else:
             steps.append(('model', self.model))

        pipeline = Pipeline(steps=steps)
        scores = self.cross_val_score(pipeline, X, Y, scoring=scorers, cv=self.cv_obj, n_jobs=None)
        return scorers,scores

    def _create_result_table(self, scores, metrics, verbose):
        averages = []
        std_devs = [] 
        for metric in metrics:
            if verbose:
                print(metric, scores['test_' + metric])
            averages.append(scores['test_' + metric].mean())
            std_devs.append(scores['test_' + metric].std())
        cv_result_table = pd.DataFrame({'metric': metrics, 'average' : averages, 
                                        'std devs': std_devs})
        if verbose:
            print(cv_result_table)
        return cv_result_table



        