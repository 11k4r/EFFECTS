import pandas as pd
import numpy as np

from tslearn.preprocessing import TimeSeriesResampler
from tsmoothie.smoother import LowessSmoother

FUNC_NAMES = {'mean': np.mean, 'min': np.min, 'max': np.max}


class Resampler:
    """
    Resampler for time series. Resample time series so that they be in equal size.

    parameters
    ----------
    sz : ind
        Size of the output time series.
        If int, then consider sz as the size of the output time series.
        If 'mean', then sz equal to the avarege of the time serieses.
        If 'min', then sz equal to the mininum of the time serieses.
        If 'max', then sz equal to the maximum of the time serieses.
    """

    def __init__(self, sz='mean'):
        self.sz = sz

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
        assert (type(self.sz) == int and self.sz > 1) or (self.sz in ['mean', 'min', 'max'])

        X = pd.DataFrame(X)
        self.col_sz = self.sz if type(self.sz) == int else \
            int(FUNC_NAMES[self.sz]([len(e) for e in X.values.reshape(-1)]))
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Transform the data so that all the samples will have the same length.

        Parameters
        ----------
        X : array-like of shape (m_ts, n_d, d_sz)
            Time series dataset to be resampled.
        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        resampler = TimeSeriesResampler(self.col_sz)
        for column in X.columns:
            X[column] = pd.Series(
                list(resampler.fit_transform(X[column]).reshape((len(X[column]), self.col_sz))))
        return X

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (m_ts, n_d, d_sz)
            Time series dataset to be resampled.
        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)


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
    def __init__(self, resampler=Resampler(), smoother=LowessAnomalyDetector()):
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
        if self.resampler != None:
            X = self.resampler.fit(X)
        if self.smoother != None:
            X = self.smoother.fit(X)
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
        if self.resampler != None:
            X = self.resampler.transform(X)
        if self.smoother != None:
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
