from ..constants import Q_ID
from ..constants import OUTPUT_DATA_PATH
from ..data_validation import check_question_df_columns, check_answers_df_columns
from ..evaluation.latent_traits_estimation import compute_error_metrics_latent_traits_estimation
from ..modules.estimators_from_text import BaseEstimatorFromText
from ..modules.latent_traits_calibration import BaseLatentTraitsCalibrator
import os
import pandas as pd
import pickle


class Text2PropsModel(object):
    def __init__(
            self, latent_traits_calibrator: BaseLatentTraitsCalibrator, estimator_from_text: BaseEstimatorFromText
    ):
        self.latent_traits_calibrator = latent_traits_calibrator
        self.n_latent_traits = latent_traits_calibrator.get_n_latent_traits()
        self.latent_traits = latent_traits_calibrator.get_name_latent_traits()
        self.estimator_from_text = estimator_from_text
        self.ground_truth_latent_traits = None

    def calibrate_latent_traits(self, df_gte):
        if self.latent_traits_calibrator is None:
            raise ValueError("No LatentTraitsCalibrator defined")
        self.ground_truth_latent_traits = self.latent_traits_calibrator.calibrate_latent_traits(df_gte)
        return self.ground_truth_latent_traits

    def train(self, df_train, df_gte=None):
        self.calibrate_latent_traits(df_gte)
        self.estimator_from_text.train(df_train, self.ground_truth_latent_traits)

    def predict(self, input_df):
        return self.estimator_from_text.predict(input_df)

    def randomized_cv_train(
            self,
            param_distributions,
            df_train: pd.DataFrame,
            df_gte: pd.DataFrame = None,
            n_iter: int = 10,
            n_jobs: int = None,
            cv=None,
            random_state: int = None,
            perform_calibration: bool = False
    ):
        """
        Performs the training with RandomizedCV of the EstimatorFromText object of the Text2PropsModel object. Then, it
        returns the scores. It can work both with already calibrated latent traits and with latent traits to be
        calibrated, it can be specified by means of the perform_calibration parameter.
        :param param_distributions: probability distribution of the parameters for the Randomized CV
        :param df_train: training dataset, it needs to have the columns specified in QUESTION_DF_COLS
        :param df_gte: dataset to perform the calibration of ground truth latent traits, it needs to have the columns
          specified in ANSWERS_DF_COLS
        :param n_iter: number of iteration of RandomizedCV
        :param n_jobs: number of jobs for parallelization
        :param cv:
        :param random_state: for reproducibility
        :param perform_calibration: boolean to select if ground truth latent have to be calibrated
        :return: The scores obtained by the best performing model in the RandomizedCV
        """
        if perform_calibration:
            check_answers_df_columns(df_gte)
        check_question_df_columns(df_train)
        if perform_calibration:
            self.calibrate_latent_traits(df_gte)
        scores = self.estimator_from_text.randomized_cv_train(
            param_distributions=param_distributions,
            df_train=df_train,
            ground_truth_latent_traits=self.ground_truth_latent_traits,
            n_iter=n_iter,
            n_jobs=n_jobs,
            cv=cv,
            random_state=random_state
        )
        return scores

    def compute_error_metrics_latent_traits_estimation(self, input_df: pd.DataFrame) -> dict:
        results = dict()
        predictions = self.predict(input_df)
        for latent_trait in self.latent_traits:
            results[latent_trait] = dict()
            y_pred = predictions[latent_trait]
            y_true = [self.ground_truth_latent_traits[latent_trait][q_id] for q_id in input_df[Q_ID].values]
            results[latent_trait] = compute_error_metrics_latent_traits_estimation(y_true, y_pred)
        return results

    def get_calibrated_latent_traits(self) -> dict:
        """
        Returns the calibrated latent traits using the get_calibrated_latent_traits of the latent traits calibrator obj
        :return:
        """
        return self.latent_traits_calibrator.get_calibrated_latent_traits()

    def store_calibrated_latent_traits(
            self,
            output_data_path: str = OUTPUT_DATA_PATH,
            output_filename: str = 'latent_traits.p'
    ):
        """
        Stores, using pickle.dump, the calibrated latent traits in the file specified as parameter (optional)
        :param output_data_path: output directory to store the data in
        :param output_filename: the name of the output file
        :return:
        """
        latent_traits = self.latent_traits_calibrator.get_calibrated_latent_traits()
        pickle.dump(latent_traits, open(os.path.join(output_data_path, output_filename), "wb"))