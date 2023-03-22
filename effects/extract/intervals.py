import numpy as np
from itertools import accumulate, groupby
from scipy.stats import norm
from scipy.stats import ks_2samp
from statistics import NormalDist

class SliceOptimizer:

    def __init__(self):
        self.intervals = []
        self.time_scores = []

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return self.intervals

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)


class FullSlicer(SliceOptimizer):

    def __init__(self):
        super().__init__()
        self.intervals = []

    def fit(self, X, y=None, **kwargs):
        series_length = len(X.values[0][0])
        series_indices = np.arange(series_length)
        self.intervals = [(a, b) for idx, a in enumerate(series_indices) for b in series_indices[idx + 1:]]
        return self


class ImplicitSlicer(SliceOptimizer):

    def __init__(self, intervals):
        super().__init__()
        assert type(intervals) == list and len(intervals) > 0
        self.intervals = intervals

    def fit(self, X, y=None, **kwargs):
        return self

class RandomSlicer(SliceOptimizer):

    def __init__(self, num_intervals=10):
        super().__init__()
        self.num_intervals = num_intervals
        self.intervals = []

    def fit(self, X, y=None, **kwargs):
        self.intervals = []
        length = len(np.array(X)[0][0])
        for i in range(self.num_intervals):
            start = np.random.randint(0, length-1)
            offset = np.random.randint(1, length-start)
            self.intervals.append((start, start+offset))
        return self


class ParametrizedSlicer(SliceOptimizer):

    def __init__(self, start=0, end=0, size=1, stride=1, method='default'):
        super().__init__()

        assert start >= 0

        self.start = start
        self.end = end
        self.size = size
        self.stride = stride
        self.method = method
        self.intervals = []

    def fit(self, X, y=None, **kwargs):
        self.intervals = []

        if self.end == 0:
            self.end = len(np.array(X)[0][0])

        if self.method == 'auto':
            self.size = self.stride = int(self.end / 10)

        self.intervals = [(i, min([i+self.size, self.end])) for i in np.arange(self.start, self.end, self.stride)]
        return self


class OptimizedSlicer(SliceOptimizer):

    def __init__(self, smooth=1e-9, union_coef=2):
        super().__init__()
        self.intervals = []
        self.smooth = smooth
        self.union_coef = union_coef

    def fit(self, X, y=None, **kwargs):
        pairs = [(a, b) for idx, a in enumerate(np.unique(y)) for b in np.unique(y)[idx + 1:]]
        length = X.values[0][0].size
        self.time_scores = np.zeros(length)

        for col in X.columns:
            col_time_scores = []
            for i in range(length):
                distibutions = {}
                for cls in np.unique(y):
                    mu1, std1 = norm.fit(np.stack(X[y == cls][col].values)[:, i])
                    distibutions[cls] = (mu1 + self.smooth, std1 + self.smooth)

                score = 0

                score_flag = False

                for pair in pairs:
                    x1 = pair[0]
                    x2 = pair[1]
                    overlap = NormalDist(mu=distibutions[x1][0], sigma=distibutions[x1][1]). \
                        overlap(NormalDist(mu=distibutions[x2][0], sigma=distibutions[x2][1]))
                    score += overlap
                    if overlap == 0:
                        score_flag = True
                if score_flag:
                    score = 0
                col_time_scores.append(1 - (score / len(pairs)))
            self.time_scores += np.array(col_time_scores)

        median = np.median(self.time_scores)

        flag_array = np.array(self.time_scores) > median

        idx = [0] + list(accumulate(sum(1 for _ in g) for _, g in groupby(flag_array)))

        start = 1 - int(flag_array[0])

        intervals = []
        for i in range(start, len(idx) - 1 - start, 2):
            intervals.append([idx[i], idx[i + 1]])

        new_intervals = []
        start = intervals[0][0]
        end = intervals[0][1]
        for interval in intervals[1:]:
            if interval[0] - end <= self.union_coef:
                end = interval[1]
            else:
                new_intervals.append([start, end])
                start = interval[0]
                end = interval[1]
        new_intervals.append([start, end])

        self.intervals = new_intervals

        return self