import numpy as np
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

    def predict(self, X):
        """
        Applies clipping to the predictions made by the base estimator.

        Args:
            X:
                The input data to make predictions on.

        Returns:
            The predictions made by the base estimator, clipped to the
            specified minimum and maximum values.
        """
        predictions = self.base_estimator.predict(X)
        return np.clip(predictions, self.min_value, self.max_value)
