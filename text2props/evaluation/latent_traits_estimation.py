from typing import Dict, List

from .constants import *
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
)


def compute_error_metrics_latent_traits_estimation(
        y_true: List[float], y_pred: List[float], metrics: List[str] = None) -> Dict[str, float]:
    dict_errors = dict()
    if metrics is None:
        metrics = [MAE, MSE, RMSE, R2, MAX_ERROR, MIN_ERROR]
    if MAE in metrics:
        dict_errors[MAE] = mean_absolute_error(y_true, y_pred)
    if MSE in metrics:
        dict_errors[MSE] = mean_squared_error(y_true, y_pred)
    if RMSE in metrics:
        dict_errors[RMSE] = np.sqrt(mean_squared_error(y_true, y_pred))
    if R2 in metrics:
        dict_errors[R2] = r2_score(y_true, y_pred)
    if MAX_ERROR in metrics:
        dict_errors[MAX_ERROR] = max_error(y_true, y_pred)
    if MIN_ERROR in metrics:
        dict_errors[MIN_ERROR] = np.min([np.abs(y_true[idx] - y_pred[idx]) for idx in range(len(y_true))])
    return dict_errors
