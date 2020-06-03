from . import BaseRegressionComponent
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


class SklearnRegressionComponent(BaseRegressionComponent):

    def __init__(self, regressor, latent_trait_range):
        self.regressor = regressor
        self.min_latent_trait = latent_trait_range[0]
        self.max_latent_trait = latent_trait_range[1]

    def train(self, x, y):
        self.regressor.fit(x, y)

    def predict(self, x):
        # TODO clean this by doing a single loop
        predictions = self.regressor.predict(x)
        predictions = [min(x, self.max_latent_trait) for x in predictions]
        predictions = [max(x, self.min_latent_trait) for x in predictions]
        return predictions

    def randomized_cv_train(self, x, y, param_distributions=None, n_iter=None, cv=None, n_jobs=None, random_state=None):
        """
        Performs the training of the regressor with RandomizedSearchCV, and returns the score obtained with the best
        estimator.
        :param x:
        :param y:
        :param param_distributions:
        :param n_iter:
        :param cv:
        :param n_jobs:
        :param random_state:
        :return:
        """
        random_search = RandomizedSearchCV(
            Pipeline(steps=[('regressor', self.regressor)], verbose=False),
            param_distributions, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state
        )
        random_search = random_search.fit(x, y)
        self.regressor = random_search.best_estimator_['regressor']
        return random_search.best_score_
