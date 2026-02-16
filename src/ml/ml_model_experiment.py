import importlib
from typing import List
import pandas as pd
from src.ml.interfaces.itrain_test_split import ITrainTestSplit
from src.ml.tarin_test_split_scikit import TrainTestSplitScikit
import src.ml.gridcv as gridcv
from src.ml.kfold_cross_validation_executor import KfoldCrossValidationExecutor
from src.ml.model_evaluator_on_split_data import ModelEvaluatorOnSplitData
from src.ml.standardise_train_test_data import standardise_data
from src.ml.evaluate_feature_importance_using_permutation_importance import EvaluateFeatureImportanceUsingPermutationImportance
from src.ml.interfaces.iml_model import IMlModel

'''if the scikit learn models does have appropriate methods they would have to be packaged up'''
class MlModelExperiment: 
    """A class to carry trivial machine learning experiments with scikit learn models (model that has fit and predict functions)
    1. train model on train set
    2. predict train set and test set
    3. find best hyperparameters using grid cv
    4. carry out k-fold cross validation

    Examples
    --------
    >>> from MachineLearningExperiment.ml_model_experiment import MlModelExperiment
    """

    def __init__(self, 
                model : IMlModel, 
                data : pd.DataFrame, 
                features : List, 
                target : str, 
                test_size = 0.2, 
                train_test_splitter : ITrainTestSplit.train_test_split = TrainTestSplitScikit().train_test_split,
                standardisation = False,
                standardise_y = False,
                random_state = 123,
                verbose=False,
                is_regression=True,
                scaler = None):
        
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.standardisation = standardisation
        self.standardise_y = standardise_y
        self.test_size = test_size
        self.train_test_split = train_test_splitter
        self.random_state = random_state
        self.standard_scaler = None
        self.verbose = verbose
        self.is_regression = is_regression
        self.standard_scaler = scaler
        self._data_prep()
        if scaler is None:
            self._train_standard_scaler()
        
    def _data_prep(self):
        self.training_data_set, self.testing_data_set = self.train_test_split(self.data, 
                                                                        test_size=self.test_size, 
                                                                        random_state=self.random_state, 
                                                                        shuffle = True)

        #self.data = pd.concat([self.training_data_set, self.testing_data_set])
        self.x = self.data[self.features]
        self.y = self.data[self.target]
        self.x_train = self.training_data_set[self.features]
        self.y_train = self.training_data_set[self.target]
        self.x_test = self.testing_data_set[self.features]
        self.y_test = self.testing_data_set[self.target]

    def _train_standard_scaler(self):
        if self.standardisation:
            self.x_train, self.standard_scaler = standardise_data(self.x_train)
            self.x_test, _ = standardise_data(self.x_test,  self.standard_scaler)
            self.x, _ = standardise_data(self.x,  self.standard_scaler)

    def  _get_best_hyperparameters(self, 
                                hyperparameter_grid, 
                                grid_cv_k_fold = 5,
                                gridCV = gridcv.gridCV):
        # x_train, y_train = shuffle.shuffle(self.x_train, self.y_train)
        model =  gridCV(self.model, hyperparameter_grid, self.x_train, self.y_train, grid_cv_split = grid_cv_k_fold, verbose=self.verbose,
                        is_regression=self.is_regression)
        if self.verbose:
            print('best parameters', model.best_params_)
            print('best score', model.best_score_)
        self.model_best_hyper_parameters = model.best_params_
        self.model_score_for_best_hyper_parameters = model.best_score_
        return model.best_params_, model.best_score_

    def optimise_hyper_parameters(self,
        hyperparameter_grid):
        self._get_best_hyperparameters(hyperparameter_grid)
        model = self.model.__class__
        self.model = model(**self.model_best_hyper_parameters)
        return self.model

    def fit_model_and_train_test_evaluation(self, model_evaluator_on_split_data = ModelEvaluatorOnSplitData, plotting = True):
        self.model.fit(self.x_train, self.y_train)
        self.model_evaluator = model_evaluator_on_split_data(model=self.model, x=self.x, y=self.y, x_train=self.x_train,
                                                            y_train=self.y_train, x_test=self.x_test, y_test=self.y_test,
                                                            is_regression=self.is_regression) # type: ignore
        self.scores = self.model_evaluator.evaluate(plotting)
        return self.scores

    def evaluate_model_cv(self, k_fold = 5, not_include_test_set_in_cv = True,
                        cross_validation_executor = KfoldCrossValidationExecutor(),
                        n_repeats = 1,
                        verbose = False):
        
        if not_include_test_set_in_cv:
            x = self.training_data_set[self.features]
            y = self.training_data_set[self.target]
        else:
            x = self.data[self.features]
            y = self.data[self.target]

        self.cv_results = cross_validation_executor.execute(x, y, self.model, self.standardisation, self.standardise_y, k_splits=k_fold,
                                                                            n_repeats=n_repeats, verbose=verbose, is_regression=self.is_regression)
        return self.cv_results

    def evaluate_model_feature_importance(self, scoring = 'r2', sort=False, feature_importance_evaluator = EvaluateFeatureImportanceUsingPermutationImportance()):
        data_ = self.x.copy()
        data_[self.target] = self.y
        importance = feature_importance_evaluator.evaluate(self.model, data_, self.features, self.target, scoring=scoring)
        feature_importance_evaluator.visualise_importance(importance, sort=sort)
        return importance