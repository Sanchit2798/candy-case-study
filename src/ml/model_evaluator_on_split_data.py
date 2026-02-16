import inspect
import pandas as pd
from src.ml.interfaces.iml_model import IMlModel
from src.ml.model_prediction_data_analyser import ModelPredictionAnalyserArrays, ModelPredictionDataAnalyser

class ModelEvaluatorOnSplitData:

    def __init__(self, model : IMlModel, x : pd.DataFrame, y : pd.DataFrame, x_train : pd.DataFrame, 
                y_train : pd.DataFrame, 
                x_test : pd.DataFrame, y_test : pd.DataFrame,
                model_prediction_data_analyser : ModelPredictionDataAnalyser = ModelPredictionAnalyserArrays,
                is_regression = True) -> None: # to do change to interface 
        self.model = model
        self.x = x
        self.y = y
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_prediction_data_analyser = model_prediction_data_analyser
        self.is_regression = is_regression

    def visualise_func(self, X : ModelPredictionDataAnalyser, plot_title):
        X.plot_prediction_data(plot_title=plot_title, is_regression= self.is_regression)

    def evaluate(self, plot = True, uncertainty_function = None):

        scores = {}
        if uncertainty_function is None:
            uncertainty_function = self.model.predict

        if 'return_std' in inspect.getfullargspec(uncertainty_function).args:
            self.y_test_pred, self.y_test_pred_std = uncertainty_function(self.x_test, return_std = True)
            self.y_train_pred, self.y_train_pred_std = uncertainty_function(self.x_train, return_std = True)
            self.y_pred, self.y_pred_std = uncertainty_function(self.x, return_std = True)
        else:
            self.y_test_pred = self.model.predict(self.x_test)
            self.y_train_pred = self.model.predict(self.x_train)
            self.y_pred = self.model.predict(self.x)
            self.y_test_pred_std, self.y_train_pred_std, self.y_pred_std = None, None, None

        self.prediction_data = {'Train':self.model_prediction_data_analyser(self.y_train.to_numpy(), self.y_train_pred, self.y_train_pred_std),
                                'Test':self.model_prediction_data_analyser(self.y_test.to_numpy(), self.y_test_pred, self.y_test_pred_std),
                                'Full':self.model_prediction_data_analyser(self.y.to_numpy(), self.y_pred, self.y_pred_std)}
        
        for key in self.prediction_data.keys():
            if plot:
                self.visualise_func(self.prediction_data[key], plot_title=key)
            scores[key] = self.prediction_data[key].get_metrics(is_regression= self.is_regression)

        return scores

