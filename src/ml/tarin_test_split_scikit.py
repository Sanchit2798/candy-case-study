import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.interfaces.itrain_test_split import ITrainTestSplit

class TrainTestSplitScikit(ITrainTestSplit):

    def __init__(self) -> None:
        super().__init__()

    def train_test_split(self, data: pd.DataFrame, test_size: float, random_state: int, shuffle: bool):
        return train_test_split(data, test_size = test_size, random_state=random_state, shuffle=shuffle)

