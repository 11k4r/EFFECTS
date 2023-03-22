import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from effects.extract.conf import SEPERATOR


class UnivariateTransform:

    def __init__(self, transform_, name="", columns=[]):

        assert callable(transform_)

        self.transform_ = transform_
        self.name = name if name != "" else transform_.__name__
        self.columns = columns

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
        X = pd.DataFrame(X)
        self.columns = X.columns
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
        X = pd.DataFrame(X)
        x_transformed = pd.DataFrame()
        for col in self.columns:
            x_transformed[SEPERATOR.join([col, self.name])] = X[col].apply(self.transform_)
        return x_transformed

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

    def set_columns(self, columns):
        self.columns = columns


class BivariateTransform:

    def __init__(self, transform_, name="", columns=[]):

        assert callable(transform_)

        self.transform_ = transform_
        self.name = name if name != "" else transform_.__name__
        self.columns = columns

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
        X = pd.DataFrame(X)
        self.columns = X.columns
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
        X = pd.DataFrame(X)
        possible_pairs = [(a, b) for idx, a in enumerate(self.columns) for b in self.columns[idx + 1:]]
        x_transformed = pd.DataFrame()
        for col1, col2 in possible_pairs:
            x_transformed[SEPERATOR.join([col1, col2, self.name])] = X.apply(
                lambda x: self.transform_(x[col1], x[col2]), axis=1)
        return x_transformed

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

    def set_columns(self, columns):
        self.columns = columns


# Univariate Transforms
identity_transform = UnivariateTransform(lambda x: x, name='identity')
diff_transform = UnivariateTransform(lambda x: np.insert(np.diff(x), 0, 0, axis=0), name='diff')
standard_transform = UnivariateTransform(lambda x: StandardScaler().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                         name='standart')
scale_transform = UnivariateTransform(lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                      name='scale')
power_transform = UnivariateTransform(lambda x: PowerTransformer().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                      name='power')
fft_transform = UnivariateTransform(lambda x: np.abs(np.fft.fft(x)), name='fft')

# Bivariate Transforms
subtraction_transform = BivariateTransform(lambda x1, x2: x1 - x2, name='subtraction')


def divide(x1, x2):
    x2[x2 == 0] = min(np.min(x2), 1) / 10e6
    return x1 / x2


divide_transform = BivariateTransform(lambda x1, x2: divide(x1, x2), name='divide')
