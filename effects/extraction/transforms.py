import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

IDENTITY = 'Identity'


class UnivariateTransform:

    def __init__(self, transform_, name=""):
        self.transform_ = transform_
        self.name = name

    def transform(self, X):
        return X.apply(lambda x: self.transform_(x))


class BivariateTransform:

    def __init__(self, transform_, name=""):
        self.transform_ = transform_
        self.name = name

    def transform(self, X):
        col1, col2 = X.columns
        return X.apply(lambda x: self.transform_(x[col1], x[col2]), axis=1)


# Univariate Transforms
identity_transform = UnivariateTransform(lambda x: x, name=IDENTITY)
diff_transform = UnivariateTransform(lambda x: np.insert(np.diff(x), 0, 0, axis=0), name='diff')
standard_transform = UnivariateTransform(lambda x: StandardScaler().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                         name='standart')
scale_transform = UnivariateTransform(lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                      name='scale')
power_transform = UnivariateTransform(lambda x: PowerTransformer().fit_transform(x.reshape(-1, 1)).reshape(-1),
                                      name='power')
fft_transform = UnivariateTransform(lambda x: np.abs(np.fft.fft(x)), name='fft')

# Bivariate Transforms
sub_transform = BivariateTransform(lambda x1, x2: x1 - x2, name='sub')


def divide(x1, x2):
    x2[x2 == 0] = np.min([np.abs(x2).min(), 1]) / 10e6
    return x1 / x2


div_transform = BivariateTransform(lambda x1, x2: divide(x1, x2), name='div')