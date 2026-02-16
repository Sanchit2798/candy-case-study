from abc import ABC, abstractmethod
import pandas as pd

class ITrainTestSplit(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train_test_split(self, data : pd.DataFrame,
                        test_size : float,
                        random_state : int,
                        shuffle : bool):
        pass

