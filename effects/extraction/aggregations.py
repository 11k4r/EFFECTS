import numpy as np


class AggregationFunction:

    def __init__(self, aggregation_, name=""):
        self.aggregation_ = aggregation_
        self.name = name

    def transform(self, X):
        return self.aggregation_(X, axis=1)


maxAgg = AggregationFunction(np.max, 'max')
minAgg = AggregationFunction(np.min, 'min')
meanAgg = AggregationFunction(np.mean, 'mean')
stdAgg = AggregationFunction(np.std, 'std')
varAgg = AggregationFunction(np.var, 'var')
medianAgg = AggregationFunction(np.median, 'median')
ptpAgg = AggregationFunction(np.ptp, 'ptp')
sumAgg = AggregationFunction(np.sum, 'sum')


def ipeak(x):
    return max(x.min(), x.max(), key=abs)


def peak(X, axis=1):
    if len(X[0]) == 1:
        return np.zeros(len(X))
    x_axis = np.arange(len(np.array(X)[0]))
    return np.array([ipeak(x) for x in X])


peakAgg = AggregationFunction(peak, 'peak')


def trend(X, axis=1):
    if len(X[0]) == 1:
        return np.zeros(len(X))
    x_axis = np.arange(len(np.array(X)[0]))
    return np.array([np.polyfit(x=x_axis, y=x, deg=1)[0] for x in X])


trendAgg = AggregationFunction(trend, 'trend')
