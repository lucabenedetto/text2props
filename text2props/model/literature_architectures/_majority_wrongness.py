from text2props.model import Text2PropsModel
from text2props.modules.latent_traits_calibration import WrongnessCalibrator
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.estimators_from_text import MajorityEstimatorFromText
from text2props.constants import WRONGNESS


class MajorityWrongnessText2PropsModel(Text2PropsModel):

    def __init__(self, known_latent_traits: dict = None):
        if known_latent_traits is not None:
            latent_traits_calibrator = KnownParametersCalibrator(latent_traits=known_latent_traits)
            if set(known_latent_traits.keys()) != {WRONGNESS}:
                raise ValueError("wrong keys in known_latent_traits dictionary")
        else:
            latent_traits_calibrator = WrongnessCalibrator()
        estimator_from_text = MajorityEstimatorFromText()
        super().__init__(latent_traits_calibrator, estimator_from_text)
