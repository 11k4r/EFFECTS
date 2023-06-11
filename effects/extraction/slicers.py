import numpy as np
import pandas as pd


class Slicer:

    def __init__(self, slicer_, name=""):
        self.slicer_ = slicer_
        self.name = name

    def get_slices(self, X, y=None):
        X = pd.DataFrame(X)
        return self.slicer_(X, y)

def random_slicer_generator(n):
    def random_slicer(X, y):
        intervals = []
        length = len(np.array(X)[0][0])
        for i in range(n):
            start = np.random.randint(0, length - 1)
            offset = np.random.randint(1, length - start)
            intervals.append((start, start + offset))
        return list(set(intervals))
    return random_slicer


def equal_slicer_generator(n):
    def equal_slicer(X, y):
        length = len(np.array(X)[0][0])
        stride = int(length / n)
        return [(i, min([i + stride, length])) for i in np.arange(0, length, stride)]
    return equal_slicer


def full_slicer(X, y):
    length = len(X.values[0][0])
    series_indices = np.arange(length)
    return [(a, b) for idx, a in enumerate(series_indices) for b in series_indices[idx + 1:]]