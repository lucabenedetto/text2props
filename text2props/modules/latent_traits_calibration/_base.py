from text2props.constants import DATA_PATH
import pandas as pd
from abc import abstractmethod
import os
import pickle


class BaseLatentTraitsCalibrator(object):

    @abstractmethod
    def __init__(self, **params):
        pass

    def get_n_latent_traits(self) -> int:
        return self.n_latent_traits

    def get_name_latent_traits(self) -> list:
        return self.name_latent_traits

    def get_calibrated_latent_traits(self) -> dict:
        return self.estimated_latent_traits

    @abstractmethod
    def calibrate_latent_traits(self, df_gte: pd.DataFrame) -> dict:
        raise NotImplementedError

    def store_calibrated_latent_traits(self, output_data_path: str = DATA_PATH, output_filename: str = 'lt.p'):
        pickle.dump(self.estimated_latent_traits, open(os.path.join(output_data_path, output_filename), "wb"))
