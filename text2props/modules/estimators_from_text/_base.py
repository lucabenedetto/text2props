from abc import ABCMeta, abstractmethod


class BaseEstimatorFromText(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def train(self, input_df, ground_truth_latent_traits):
        raise NotImplementedError

    @abstractmethod
    def randomized_cv_train(self, **params):
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_df) -> dict:
        raise NotImplementedError
