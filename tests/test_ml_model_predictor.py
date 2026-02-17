from ast import Set
import unittest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import sys
import pandas as pd
from src.ml.ml_model_predictor import MlModelPredictor

sys.path.append('..')

class TestMlModelPredictor(unittest.TestCase):

    data = pd.DataFrame({'Feature 1':[1,2,3,4,5,6,7,8],
                         'Feature 2':[10, 20, 30, 40, 50, 60, 70, 80],
                         'Target': [11, 22, 33, 44, 55, 66, 77, 88]})
    features = ['Feature 1', 'Feature 2']
    target = ['Target']

    def test_predict_without_return(self):
        ## ARRANG
        scaler = StandardScaler()
        data_std = scaler.fit_transform(self.data[self.features])
        lr = LinearRegression()
        lr.fit(data_std, self.data['Target'])
        ## ACT
        prediction = MlModelPredictor().predict(self.data, self.features,
                                   lr, scaler)
        ## ASSERT
        assert len(prediction) == 8

    def test_predict_with_return(self):
        scaler = StandardScaler()
        data_std = scaler.fit_transform(self.data[self.features])
        gpr = GaussianProcessRegressor()
        gpr.fit(data_std, self.data['Target'])
        prediction, prediction_std = MlModelPredictor().predict(self.data, self.features,
                                   gpr, scaler, return_std=True)
        assert len(prediction) == 8
        
        assert len(prediction_std) == 8