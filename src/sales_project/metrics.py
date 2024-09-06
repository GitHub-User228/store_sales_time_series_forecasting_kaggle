from typing import Literal

import numpy as np
from sklearn.metrics import *


def mean_cv_scores(cv_res: dict, ndigits: int = 4) -> dict:
    """
    Calculates the mean of the cross-validation's scores in a dictionary
    for regression task.

    Args:
        cv_res (dict):
            A dictionary containing the cross-validation scores.
        ndigits (int):
            The number of decimal places to round the mean scores to.
            Defaults to 4.

    Returns:
        dict:
            A dictionary containing the mean cross-validation scores
            for regression task.
    """

    cv_res_mean = {}

    for key, value in cv_res.items():
        cv_res_mean[key] = round(value.mean(), ndigits)
        if "neg_" in key:
            cv_res_mean[key.replace("neg_", "")] = cv_res_mean[key] * -1
            del cv_res_mean[key]

    return cv_res_mean


METRICS = Literal["MAE", "RMSE", "RMSLE", "R2"]


METRICS_FUNCTIONS = {
    "MAE": mean_absolute_error,
    "RMSE": root_mean_squared_error,
    "RMSLE": root_mean_squared_log_error,
    "R2": r2_score,
}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[METRICS] = ["MAE", "RMSE", "RMSLE", "R2"],
) -> dict:
    """
    Evaluates the performance of a machine learning model by calculating
    various metrics on the actual and predicted target values.

    Args:
        y_true (np.ndarray):
            The actual target values.
        y_pred (np.ndarray):
            The predicted target values.
        metrics (list[METRICS]):
            The list of metrics to calculate.

    Returns:
        dict: A dictionary containing some of the following performance
            metrics:
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Squared Error
            - RMSLE: Root Mean Squared Log Error
            - R2: Coefficient of Determination
    """
    values = {}
    for metric in metrics:
        values[metric] = float(METRICS_FUNCTIONS[metric](y_true, y_pred))
    return values
