from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def standardise_data(data: pd.DataFrame, scaler = None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data)
    if (type(data) == pd.DataFrame):
        scaled_data = pd.DataFrame( scaler.transform(data), columns=data.columns, index=data.index )
    elif (type(data) == np.ndarray) :
        scaled_data = scaler.transform(data)
    else:
        raise ValueError(("Data input type not supported"))
    return scaled_data, scaler