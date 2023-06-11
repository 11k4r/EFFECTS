import pandas as pd
import numpy as np

class Preprocessor:
    """
    EFFECTS Preprocessor.

    Parameters
    ----------
    resampler : object, default=Resampler()
        time-series resampler
        will result in a smoother curve.
    smoother : object, default=LowessAnomalyDetector()
        Anomaly detector
    """

    def __init__(self, resampler=None, smoother=None):
        self.resampler = resampler
        self.smoother = smoother

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
        if self.resampler is not None:
            self.resampler.fit(X)
        if self.smoother is not None:
            self.smoother.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Preprocess time-series data.

        Parameters
        ----------
        X : array-like of shape (m_ts, n_d, d_sz)
            Time series dataset to be resampled.
        Returns
        -------
        numpy.ndarray
            preprocessed time series dataset.
        """
        if self.resampler is not None:
            X = self.resampler.transform(X)
        if self.smoother is not None:
            X = self.smoother.transform(X)
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
            Proprocessed time series dataset.
        """
        return self.fit(X).transform(X)
