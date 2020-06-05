from ._base import BaseEstimatorFromText
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.regression import RegressionModule
from text2props.constants import Q_ID
import pandas as pd


class FeatureEngAndRegressionEstimatorFromText(BaseEstimatorFromText):

    def __init__(self, pipelines: dict):
        super().__init__()
        self.pipelines = pipelines  # one pipeline for each latent trait

    def train(self, df_train: pd.DataFrame, ground_truth_latent_traits: dict):
        """
        Trains the EstimatorFromText object, training all the pipelines contained in the object.
        :param df_train:
        :param ground_truth_latent_traits:
        :return:
        """
        for latent_trait in self.pipelines.keys():
            local_y = [ground_truth_latent_traits[latent_trait][q_id] for q_id in df_train[Q_ID].values]
            self.pipelines[latent_trait].train(df_train, local_y)

    def predict(self, input_df: pd.DataFrame):
        """
        Performs the prediction. The returned object is a dictionary whose keys are the names of the latent traits.
        :param input_df:
        :return:
        """
        predictions = dict()
        for latent_trait in self.pipelines.keys():
            predictions[latent_trait] = self.pipelines[latent_trait].predict(input_df)
        return predictions

    def randomized_cv_train(
            self,
            param_distributions: dict,
            df_train: pd.DataFrame,
            ground_truth_latent_traits: dict,
            n_iter: int = 10,
            n_jobs: int = None,
            cv: int = None,
            random_state: int = None,
    ):
        """
        Trains the EstimatorFromText object with RandomizedCV, bu training all the pipelines, and returns the scores so
        that they can be compared with the results obtained with other models.
        :param param_distributions:
        :param df_train:
        :param ground_truth_latent_traits:
        :param n_iter:
        :param n_jobs:
        :param cv:
        :param random_state:
        :return:
        """
        scores = []
        for latent_trait in self.pipelines.keys():
            local_y = [ground_truth_latent_traits[latent_trait][q_id] for q_id in df_train[Q_ID].values]
            scores.append(
                self.pipelines[latent_trait].randomized_cv_train(
                    param_distributions=param_distributions[latent_trait],
                    df_train=df_train,
                    y_train=local_y,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    cv=cv,
                    random_state=random_state
                )
            )
        return scores


class FeatureEngAndRegressionPipeline(object):
    def __init__(self, feature_engineering: FeatureEngineeringModule, regression: RegressionModule):
        self.feat_eng_module = feature_engineering
        self.regression_module = regression

    def train(self, input_df, y):
        partial_results = self.feat_eng_module.fit_transform(input_df)
        self.regression_module.train(partial_results, y)

    def predict(self, x):
        partial_results = self.feat_eng_module.transform(x)
        return self.regression_module.predict(partial_results)

    def randomized_cv_train(self, param_distributions, df_train, y_train, n_iter, n_jobs, cv, random_state):
        partial_results = self.feat_eng_module.fit_transform(df_train)
        score = self.regression_module.randomized_cv_train(
            partial_results, y_train, param_distributions=param_distributions, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
            random_state=random_state
        )
        return score
