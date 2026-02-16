import pickle
from typing import List
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import os
from src.ml.hyperparameter_optimisers import optimise_hyper_parameters_gpr, optimise_hyper_parameters_xgboost
from src.ml.kfold_cross_validation_executor import KfoldCrossValidationExecutor
from src.ml.ml_model_experiment import MlModelExperiment
from src.ml.ml_model_experiment_utilities import plot_save_ml_experiment_train_test_results_dfs


class TrainEvaluateSaveMlModel:
    ''''Train a machine learning model
    Underneath a machine learning experiment is run
    which train a model on training data 
    test size defines the ratio of the data reserved for testing later on
    best hyperparameters for the models are found using Grid CV'''

    successful_load : bool = False

    def __init__(self, features : List[str], target : str, 
                data : pd.DataFrame,
                test_size = 0.1,
                standardisation = True,
                standardise_y = False,
                model_file_name = None,
                scaler_file_name = None,
                prediction_data_file_name = None,
                cross_validation_executors = [KfoldCrossValidationExecutor()],
                optimise_hyper_parameters = True,
                param_grid = None,
                train_test_split = train_test_split,
                k_fold = 5,
                n_repeats_for_cv=1,
                cv_result_file_name = None,
                combined_model_file_name = None,
                plotting = True,
                verbose = False,
                load_files = False):

        self.features = features
        self.target = target
        self.data = data
        self.n_repeats_for_cv = n_repeats_for_cv
        self.test_size = test_size
        self.standardisation = standardisation
        self.standardise_y = standardise_y
        self.verbose = verbose

        self.model_file_name = model_file_name
        self.scaler_file_name = scaler_file_name
        self.prediction_data_file_name = prediction_data_file_name
        self.cross_validation_executors = cross_validation_executors
        self.optimise_hyper_parameters = optimise_hyper_parameters
        self.param_grid = param_grid
        self.train_test_split = train_test_split
        self.k_fold = k_fold
        self.cv_result_file_name = cv_result_file_name
        self.combined_model_file_name = combined_model_file_name
        self.plotting = plotting
        self.load_files = load_files

    def _run_ml_experiment(self, ml_model, optimisation_fucn = None,
                           is_regression=True):
        self.ml_model_experiment = MlModelExperiment(model=ml_model, 
                                                    data=self.data, 
                                                    features=self.features,
                                                    target = self.target,
                                                    standardisation=self.standardisation,
                                                    standardise_y=self.standardise_y,
                                                    test_size=self.test_size,
                                                    train_test_splitter=self.train_test_split,
                                                    verbose=self.verbose,
                                                    is_regression=is_regression)
        
        # optimising hyper paramters
        if self.optimise_hyper_parameters:
            if optimisation_fucn is not None:
                self.ml_model_experiment = optimisation_fucn(self.ml_model_experiment, self.param_grid)
            elif self.param_grid is not None:
                self.ml_model_experiment.optimise_hyper_parameters(self.param_grid)
            else:
                pass

        # cross validation
        for cv in self.cross_validation_executors:
            self.ml_model_experiment.evaluate_model_cv(k_fold = self.k_fold, cross_validation_executor=cv, n_repeats=self.n_repeats_for_cv, verbose=self.verbose)

        # test train split
        self.ml_model_experiment.fit_model_and_train_test_evaluation(plotting = self.plotting)

        plot_save_ml_experiment_train_test_results_dfs(self.ml_model_experiment, self.prediction_data_file_name, self.plotting) ## prediction dta filr should be saved by this class
        
        return self.ml_model_experiment


    def _save_ml_experiment_model_files_seperately(self, ml_model_experiment:MlModelExperiment):
        if self.cv_result_file_name is not None:
            ml_model_experiment.cv_results.to_csv(self.cv_result_file_name, index=False)
        
        if self.model_file_name is not None:
            joblib.dump(ml_model_experiment.model, open(self.model_file_name, 'wb'))
        if self.scaler_file_name is not None:
            joblib.dump(ml_model_experiment.standard_scaler, open(self.scaler_file_name, 'wb'))

        return ml_model_experiment

    def _save_ml_experiment_model(self, ml_model_experiment:MlModelExperiment):
        if not self.successful_load:
            pipeline = Pipeline([('scaler', ml_model_experiment.standard_scaler),
                                ('model', ml_model_experiment.model)])
            
            if self.combined_model_file_name is not None:
                joblib.dump(pipeline, self.combined_model_file_name)

            self._save_ml_experiment_model_files_seperately(ml_model_experiment)
        return ml_model_experiment

    def base_line_model(self, strategy = 'mean'):
        model = DummyRegressor(strategy=strategy)
        return self._save_ml_experiment_model(self._run_ml_experiment(model, is_regression=True))

    def linear_regression_model(self):
        ml_model = LinearRegression()
        return self._save_ml_experiment_model(self._run_ml_experiment(ml_model,
                                                                      is_regression=True))

    def bayesian_ridge_model(self):
        ml_model = linear_model.BayesianRidge()
        return self._save_ml_experiment_model(self._run_ml_experiment(ml_model,
                                                                      is_regression=True))
    def xg_boost_model(self):
        ml_model = xgboost.XGBRegressor()
        optimisation_fucn = optimise_hyper_parameters_xgboost
        return self._save_ml_experiment_model(self._run_ml_experiment(ml_model, optimisation_fucn,
                                                                      is_regression=True))
    
    def gpr_model(self):
        ml_model = GaussianProcessRegressor()
        optimisation_fucn = optimise_hyper_parameters_gpr
        return self._save_ml_experiment_model(self._run_ml_experiment(ml_model, optimisation_fucn,
                                                                      is_regression=True))

    def run(self, model_type):
        if model_type == 'linear_reg':
            return self.linear_regression_model()
        if model_type == 'base_line':
            return self.base_line_model()
        if model_type == 'bayesian_ridge':
            return self.bayesian_ridge_model()
        if model_type == 'xgboost':
            return self.xg_boost_model()
        if model_type == 'gpr':
            return self.gpr_model()
        else:
            raise Exception("Model type not supprted. Valid model types are: \n \
                                                                             Gaussian Process Regressor: gpr \n  \
                                                                             eXtreme Gradient Boosting: xgboost \n \
                                                                             Linear Regression: linear_reg \n \
                                                                             Bayesian Ridge Regression: bayesian_ridge \n \
                                                                             Baseline Model: base_line")
        
    def run_with_my_model(self, ml_model):
        return self._save_ml_experiment_model(self._run_ml_experiment(ml_model))