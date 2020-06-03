import pandas as pd
from text2props.constants import (
    DIFFICULTY,
    DISCRIMINATION,
    DIFFICULTY_RANGE,
    DEFAULT_DISCRIMINATION,
)
from .utils import irt_estimation
from . import BaseLatentTraitsCalibrator


class IRTCalibrator(BaseLatentTraitsCalibrator):

    def __init__(
            self,
            difficulty_range=DIFFICULTY_RANGE,
            discrimination_range=(DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION),
    ):
        if discrimination_range[0] == discrimination_range[1]:
            self.n_latent_traits = 1
            self.name_latent_traits = [DIFFICULTY]
        else:
            self.n_latent_traits = 2
            self.name_latent_traits = [DIFFICULTY, DISCRIMINATION]
        self.estimated_latent_traits = dict()
        self.difficulty_range = difficulty_range
        self.discrimination_range = discrimination_range

    def get_n_latent_traits(self):
        return self.n_latent_traits

    def get_name_latent_traits(self):
        return self.name_latent_traits

    def get_calibrated_latent_traits(self) -> dict:
        return self.estimated_latent_traits

    def calibrate_latent_traits(self, df_gte: pd.DataFrame) -> dict:
        estimated_latent_traits = irt_estimation(df_gte, self.difficulty_range, self.discrimination_range)
        self.estimated_latent_traits[DIFFICULTY] = estimated_latent_traits[DIFFICULTY]
        if self.n_latent_traits == 2:
            self.estimated_latent_traits[DISCRIMINATION] = estimated_latent_traits[DISCRIMINATION]
        return self.estimated_latent_traits
