from abc import ABC, abstractmethod

class IMlModel:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def get_params():
        pass

    @abstractmethod
    def set_params():
        pass