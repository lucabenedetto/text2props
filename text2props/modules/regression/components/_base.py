from abc import ABCMeta, abstractmethod


class BaseRegressionComponent(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def randomized_cv_train(self, x, y, **params):
        raise NotImplementedError
