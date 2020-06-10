from abc import ABCMeta, abstractmethod
from typing import Dict
import pandas as pd


class BaseEstimatorFromText(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def train(self, input_df: pd.DataFrame, ground_truth_latent_traits: Dict[str, Dict[str, float]]):
        raise NotImplementedError

    @abstractmethod
    def randomized_cv_train(self, **params):
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_df) -> dict:
        raise NotImplementedError
