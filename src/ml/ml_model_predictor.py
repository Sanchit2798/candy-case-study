import pandas as pd
from typing import List
from src.ml.interfaces.iml_model import IMlModel
from src.ml.interfaces.iscaler import IScaler

class MlModelPredictor:

    def __init__(self) -> None:
        pass

    def predict(self, data : pd.DataFrame, features : List[str], model : IMlModel, scaler:IScaler = None,
                return_std=False):
        if scaler is not None:
            predicted_data_stdandardised = scaler.transform(data[features])
        else:
            predicted_data_stdandardised = data[features]

        if return_std:
            predicted_data, predicted_data_std = model.predict(predicted_data_stdandardised, return_std = True)
            return predicted_data, predicted_data_std
        else:
            predicted_data = model.predict(predicted_data_stdandardised)
            return predicted_data