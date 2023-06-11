import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler

RESAMPLER_FUNC_NAMES = {'mean': np.mean, 'min': np.min, 'max': np.max}

class Resampler:
    """
    Resampler for time series. Resample time series so that they be in equal size.

    parameters
    ----------
    sz : int
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
            int(RESAMPLER_FUNC_NAMES[self.sz]([len(e) for e in X.values.reshape(-1)]))
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



