from effects.feature_extraction.transforms import *
import numpy as np

UNIVARIATE_TRANSFORMS = [diff]
BIVARIATE_TRANSFORMS = [subtraction]
AGGREGATION_FUNCTIONS = [np.max, np.min, np.mean, np.var]

M = 10e6