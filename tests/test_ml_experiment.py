from re import X
import unittest
from unittest.mock import call, patch, MagicMock
import mock
import pandas as pd
import sys
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from src.ml.ml_model_experiment import MlModelExperiment


sys.path.append('..')

class Test_MLExperiment(unittest.TestCase):
    """
    Class for testing the Test_MLExperiment
    """
    data = pd.read_csv('./tests/test_data/test_data.csv')
    
    features = ['X1', 'X2']
    target = ['Y']

    def test_data_prep(self):
        model = 'model'
        with mock.patch('src.ml.ml_model_experiment.standardise_data') as mock_standardise_data:
            mock_standardise_data.side_effect = [(1, 'scaler'),(2, 'scaler'),(3, 'scaler')]
            test_experiment = MlModelExperiment(model, self.data, self.features, self.target, standardisation=True)
            x_train = test_experiment.training_data_set[self.features]
            x_test = test_experiment.testing_data_set[self.features]
            x = test_experiment.data[self.features]
            args, kwargs = mock_standardise_data.call_args_list[0]
            pd.testing.assert_frame_equal(
                args[0],
                x_train
            )
            args, kwargs = mock_standardise_data.call_args_list[1]
            pd.testing.assert_frame_equal(
                args[0],
                x_test
            )
            assert  args[1] == 'scaler'
            args, kwargs = mock_standardise_data.call_args_list[2]
            pd.testing.assert_frame_equal(
                args[0],
                x
            )
            assert  args[1] == 'scaler'
            assert test_experiment.x_train == 1
            assert test_experiment.x_test == 2
            assert test_experiment.x == 3
            
            pd.testing.assert_frame_equal(
                test_experiment.y,
                self.data[self.target]
            )
            

    def test_get_best_hyperparameters(self):
        model = 'model'
        test_experiment = MlModelExperiment(model, self.data, self.features, self.target)
        mock_optimised_model = MagicMock()
        mock_optimised_model.best_params_ = 'best params'
        mock_optimised_model.best_score_ = '1.0'
        mock_grid_cv = MagicMock(return_value = mock_optimised_model)
        actual_best_params, actual_best_score = test_experiment._get_best_hyperparameters(None, gridCV = mock_grid_cv)
        mock_grid_cv.assert_called_once_with(model, None, test_experiment.x_train, test_experiment.y_train, 
                                                    grid_cv_split = 5, verbose=False, is_regression=True) 
        assert mock_optimised_model.best_params_ == actual_best_params
        assert mock_optimised_model.best_score_ == actual_best_score

    def test_fit_on_train_data(self):
        model = MagicMock()
        model.fit = MagicMock(return_value = True)
        test_experiment = MlModelExperiment(model, self.data, self.features, self.target)
        test_experiment.fit_model_and_train_test_evaluation(model_evaluator_on_split_data=MagicMock())
        model.fit.assert_called_once_with(test_experiment.x_train, test_experiment.y_train)

    def test_evaluate_model_performnce_on_split_data(self):
        model = MagicMock()
        test_experiment = MlModelExperiment(model, self.data, self.features, self.target)
        mock_model_evaluator_on_split_data = MagicMock()
        mock_model_evaluator_on_split_data.evaluate = MagicMock()
        test_experiment.fit_model_and_train_test_evaluation(model_evaluator_on_split_data=mock_model_evaluator_on_split_data)
        mock_model_evaluator_on_split_data.assert_called_once()

        mock_model_evaluator_on_split_data.call_args == call(model = model, x = test_experiment.x, y = test_experiment.y, x_train = test_experiment.x_train, 
                                                            y_train = test_experiment.y_train, x_test = test_experiment.x_test, y_test = test_experiment.y_test)
        mock_model_evaluator_on_split_data().evaluate.assert_called_once()
        
    
    def test_evaluate_model_cv(self):
        model = 'model'
        test_experiment = MlModelExperiment(model, self.data, self.features, self.target)
        expected_result = pd.DataFrame([1,2]) 
        mock_cv_exectuor = MagicMock()
        mock_cv_exectuor.execute = MagicMock(return_value = expected_result)
        actual_result = test_experiment.evaluate_model_cv(cross_validation_executor= mock_cv_exectuor)
        pd.testing.assert_frame_equal(
            actual_result,
            expected_result
        )
        
        args, kwargs = mock_cv_exectuor.execute.call_args_list[0]
        pd.testing.assert_frame_equal(
            args[0],
            test_experiment.x_train
        )
        pd.testing.assert_frame_equal(
            args[1],
            test_experiment.y_train
        )

    def test_optimise_hyper_parameters(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor()
        kernels = []
        for ck1 in range(1,2):   #tried [0.01, 0.1, 1,2]
            for rbf in range(5,6): # tried [0.01,0.1,1,2]
                for wk in range(1,2): # tried [0.01,0.1,1,2]
                    kernel_ = ConstantKernel(ck1) + RBF(rbf)  + WhiteKernel(wk)
                    kernels.append(kernel_)
                    
        gpr_param_grid = { 
                            # "kernel": kernels,
                            'random_state':[123], 
                            'n_restarts_optimizer':[1, 10],
                            'alpha':[0.1], 
                            'normalize_y':[True, False]
                        }
        self.data = pd.concat([self.data, self.data, self.data])
        test_experiment = MlModelExperiment(model, self.data, self.features, self.target)
        test_experiment.optimise_hyper_parameters(gpr_param_grid)
        test_experiment.model.get_params()['alpha'] == 0.1
        test_experiment.model.get_params()['n_restarts_optimizer'] == 10

