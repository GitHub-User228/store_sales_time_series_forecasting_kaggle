import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin


class ClippingRegressor(BaseEstimator, RegressorMixin):
    """
    A regressor that clips the output of a base estimator to a
    specified range.

    This class wraps a base estimator and ensures that the predictions
    made by the estimator are clipped to a specified minimum and maximum
    value. This can be useful when the base estimator may produce values
    outside of a desired range.
    """

    def __init__(
        self,
        base_estimator,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        """
        Initializes a `ClippingRegressor` instance with the provided base
        estimator and optional minimum and maximum values.

        Args:
            base_estimator:
                The base estimator to be wrapped by the `ClippingRegressor`.
            min_value (float, optional):
                The minimum value to which the predictions will be clipped.
                If not provided, no minimum clipping will be applied.
            max_value (float, optional):
                The maximum value to which the predictions will be clipped.
                If not provided, no maximum clipping will be applied.
        """
        self.base_estimator = base_estimator
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y):
        """
        Fits the base estimator on the provided X and y data, and
        returns the ClippingRegressor instance.

        Args:
            X:
                The input data to fit the base estimator on.
            y:
                The target values to fit the base estimator on.

        Returns:
            The ClippingRegressor instance, after fitting the base
            estimator.
        """
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """
        Applies clipping to the predictions made by the base estimator.

        Args:
            X:
                The input data to make predictions on.

        Returns:
            The predictions made by the base estimator, clipped to the
            specified minimum and maximum values.
        """
        predictions = np.clip(
            self.base_estimator.predict(X), self.min_value, self.max_value
        )
        return predictions


class ClippingSequentialRegressor(BaseEstimator, RegressorMixin):
    """
    A regressor that clips the output of a base estimator to a
    specified range.

    This class wraps a base estimator and ensures that the predictions
    made by the estimator are clipped to a specified minimum and maximum
    value. This can be useful when the base estimator may produce values
    outside of a desired range.
    """

    def __init__(
        self,
        base_estimator,
        timestamp_col: str,
        target_col: str,
        auxiliary_cols: list[str],
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        """
        Initializes a `ClippingRegressor` instance with the provided base
        estimator and optional minimum and maximum values.

        Args:
            base_estimator:
                The base estimator to be wrapped by the `ClippingRegressor`.
            min_value (float, optional):
                The minimum value to which the predictions will be clipped.
                If not provided, no minimum clipping will be applied.
            max_value (float, optional):
                The maximum value to which the predictions will be clipped.
                If not provided, no maximum clipping will be applied.
        """
        self.base_estimator = base_estimator
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.auxiliary_cols = [f"auxiliary__{col}" for col in auxiliary_cols]
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y):
        """
        Fits the base estimator on the provided X and y data, and
        returns the ClippingRegressor instance.

        Args:
            X:
                The input data to fit the base estimator on.
            y:
                The target values to fit the base estimator on.

        Returns:
            The ClippingRegressor instance, after fitting the base
            estimator.
        """
        self.base_estimator.fit(
            X.drop(columns=self.auxiliary_cols).values.astype("float32"), y
        )
        return self


def sequential_predictions(
    pipeline: Pipeline,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    timestamp_col: str,
    target_col: str,
) -> pd.DataFrame:

    rolling_expanding_cols = [
        k
        for k in data.columns
        if any([v in k for v in ["rolling", "expanding"]]) and "lag" not in k
    ]
    cols_to_lag = [k for k in data.columns if "lag" in k]
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    for it, date in enumerate(tqdm(date_range, total=len(date_range))):

        data.loc[(data[timestamp_col] == date), target_col] = pipeline.predict(
            data[(data[timestamp_col] == date)]
        )

        if it < len(date_range) - 1:

            for col in rolling_expanding_cols:
                if "window" in col:
                    window = int(
                        re.findall(r"window\.\d+", col)[0].split(".")[1]
                    )
                    col2 = re.sub(r"window\.\d+", "", col)
                    col2 = re.sub(r"\.rolling\.mean\.", "", col2)
                    data.loc[data[timestamp_col] == date, col] = (
                        data.loc[
                            (
                                data[timestamp_col]
                                >= date - pd.Timedelta(days=window - 1)
                            )
                            & (data[timestamp_col] <= date)
                        ]
                        .groupby(["store_nbr", "family"])[col2]
                        .mean()
                        .values
                    )
                else:
                    col2 = re.sub(r"\.expanding\.mean", "", col)
                    data.loc[data[timestamp_col] == date, col] = (
                        data.loc[data[timestamp_col] <= date]
                        .groupby(["store_nbr", "family"])[col2]
                        .mean()
                        .values
                    )

            for col in cols_to_lag:
                lag = int(re.findall(r"lag\.\d+", col)[0].split(".")[1])
                col_to_lag = re.sub(r"\.lag\.\d+", "", col)
                data.loc[data[timestamp_col] == date_range[it + 1], col] = (
                    data.loc[
                        data[timestamp_col]
                        == date_range[it + 1] - pd.Timedelta(days=lag),
                        col_to_lag,
                    ].values
                )

    return data
