import os
import pandas as pd
from typing import List
from src.ml.interfaces.iml_model import IMlModel
from src.ml.interfaces.iscaler import IScaler
import pickle
from joblib import dump, load

class MLModelReader:

    def __init__(self) -> None:
        pass

    def read_sklearn_model(self, path : str, scaler_path=None) -> IMlModel:

        _, ext1 = os.path.splitext(path)
        if scaler_path is not None:
            _, ext2 = os.path.splitext(scaler_path)
            assert ext1 == ext2, "Both model and scaler should be of same file type"
        if ext1 == ".pkl":
            model, scaler = self.read_sklearn_model_using_pickle(path, scaler_path)
        elif ext1 == ".joblib":
            model, scaler = self.read_sklearn_model_using_joblib(path, scaler_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext1}")
        
        return model, scaler

    def read_sklearn_model_using_pickle(self, path : str, scaler_path = None) -> IMlModel:
        """
        Reads in a pickle file containing a sklearn model
        """
        scaler = None
        try:
            with open(path, 'rb') as model_file:
                model = pickle.load(model_file)
        except:
            raise Exception("invalid model")
        
        if scaler_path:
            try:
                with open(scaler_path, 'rb') as scaler_file:
                    scaler = pickle.load(scaler_file)
            except:
                raise Exception("invalid scaler")

        return model, scaler
    
    def read_sklearn_model_using_joblib(self, path : str, scaler_path = None) -> IMlModel:
        """
        Reads in a pickle file containing a sklearn model
        """
        scaler = None
        try:
            model = load(path)
        except:
            raise Exception("invalid model")
        
        if scaler_path:
            try:
                scaler = load(scaler_path)
            except:
                raise Exception("invalid scaler")
            
        return model, scaler