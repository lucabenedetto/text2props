from ._base import BaseEstimatorFromText
from text2props.constants import Q_ID
import numpy as np


class MajorityEstimatorFromText(BaseEstimatorFromText):
    """
    Very simple EstimatorFromText object. It does not really work on the texts, it only considers the average value of
    latent traits in the training set and assigns that value to all the questions in the test set.
    """

    def __init__(self):
        super().__init__()
        self.latent_traits = dict()

    def train(self, df_train, ground_truth_latent_traits):
        for latent_trait in ground_truth_latent_traits.keys():
            self.latent_traits[latent_trait] = np.mean(
                [ground_truth_latent_traits[latent_trait][q_id] for q_id in df_train[Q_ID].values]
            )

    def predict(self, input_df):
        predictions = dict()
        for latent_trait in self.latent_traits.keys():
            predictions[latent_trait] = [self.latent_traits[latent_trait]]*len(input_df.index)
        return predictions

    def randomized_cv_train(self):
        raise NotImplementedError
