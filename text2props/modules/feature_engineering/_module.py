from scipy.sparse import (
    hstack,
    coo_matrix,
)
from .components import BaseFeatEngComponent


class FeatureEngineeringModule(object):

    def __init__(self, components=None):
        if components is None:
            self.components = []
        else:
            if type(components) != list:
                raise ValueError("components should be a list")
            self.components = components

    def add_component(self, component: BaseFeatEngComponent):
        """
        Add a component to a existing feature engineering module.
        :param component:
        :return:
        """
        self.components.append(component)

    def add_list_components(self, list_components: list):
        """
        Add components to a existing feature engineering module.
        :param list_components:
        :return:
        """
        for component in list_components:
            self.components.append(component)

    def fit_transform(self, input_df) -> coo_matrix:
        """
        Call the fit_transform method on all the components of the module and stack the results.
        :param input_df:
        :return:
        """
        partial_results = []
        for component in self.components:
            partial_results.append(component.fit_transform(input_df))
        return hstack(partial_results)

    def transform(self, input_df) -> coo_matrix:
        """
        Call the transform method on all the components of the module and stack the results.
        :param input_df:
        :return:
        """
        partial_results = []
        for component in self.components:
            partial_results.append(component.transform(input_df))
        return hstack(partial_results)
