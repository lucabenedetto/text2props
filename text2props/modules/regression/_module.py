from typing import List, Dict

import numpy as np
from scipy.sparse import coo_matrix

from .components import BaseRegressionComponent


class RegressionModule(object):

    def __init__(self, components: List[BaseRegressionComponent] = None):
        if components is None:
            self.components = []
        else:
            self.components = components

    def add_component(self, component: BaseRegressionComponent):
        self.components.append(component)

    def add_list_components(self, list_components: List[BaseRegressionComponent]):
        for component in list_components:
            self.components.append(component)

    def train(self, x: coo_matrix, y: List[float]):
        for component in self.components:
            component.train(x, y)

    def predict(self, x: coo_matrix):
        partial_results = []
        for component in self.components:
            partial_results.append(component.predict(x))
        return np.mean(partial_results, axis=0)

    def randomized_cv_train(
            self,
            x_train: coo_matrix,
            y_train: List[float],
            param_distributions: List[Dict[str, List[float]]],
            n_iter: int,
            cv: int,
            n_jobs: int,
            random_state: int
    ) -> float:
        """
        Performs the training with randomized cross validation of all the components in the Regression module. For each
        component, computes the R2 score of the best performing configuration. In the end, it returns the R2 score
        obtained as the average of the R2 scores of all the components.
        :param x_train:
        :param y_train:
        :param param_distributions:
        :param n_iter:
        :param cv:
        :param n_jobs:
        :param random_state:
        :return:
        """
        scores = []
        for idx, component in enumerate(self.components):
            component_score = component.randomized_cv_train(
                x_train,
                y_train,
                param_distributions=param_distributions[idx],
                n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
            scores.append(component_score)
        return np.float(np.mean(scores))
