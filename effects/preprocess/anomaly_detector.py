import pandas as pd
import numpy as np

from tsmoothie.smoother import LowessSmoother


class LowessAnomalyDetector:
    """
    Anomaly detector using LowessSmoother as implemented in tsmoothie package.

    Parameters
    ----------
    smooth_fraction : float, default=0.1
        Between 0 and 1. The smoothing span. A larger value of smooth_fraction
        will result in a smoother curve.
    iterations : int, default=1
        Between 1 and 6. The number of residual-based reweightings to perform.
    """

    def __init__(self, smooth_fraction=0.1, iterations=1):
        self.smooth_fraction = smooth_fraction
        self.iterations = iterations

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.
        Parameters
        ----------
        X
        Ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Denoising and smoothing time-series data.

        Parameters
        ----------
        X : array-like of shape (m_ts, n_d, d_sz)
            Time series dataset to be resampled.
        Returns
        -------
        numpy.ndarray
            smoothed time series dataset.
        """
        X = pd.DataFrame(X)
        for column in X.columns:
            smoother = LowessSmoother(smooth_fraction=self.smooth_fraction, iterations=self.iterations)
            smoother.smooth(np.stack(X[column].values))
            low, up = smoother.get_intervals('prediction_interval')
            X[column] = pd.Series(list(np.clip(np.stack(X[column].values), low, up)))
        return X

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (m_ts, n_d, d_sz)
            Time series dataset to be denoise.
        Returns
        -------
        numpy.ndarray
            Denoised time series dataset.
        """
        return self.fit(X).transform(X)


