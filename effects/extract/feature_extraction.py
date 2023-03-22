import numpy as np
import pandas as pd

#TODO seperate fit and transform correctly


from effects.extract.transforms import *
from effects.extract.intervals import OptimizedSlicer
from effects.extract.aggregations import MaxAgg, MinAgg, MeanAgg, PtpAgg, TrendAgg, StdAgg, SumAgg, MedianAgg, VarAgg, PeakAgg
from effects.extract.conf import SEPERATOR
from  effects.preprocessing.preprocess import Preprocessor

U_TRANSFORMS = [identity_transform, diff_transform, standard_transform, scale_transform, power_transform, fft_transform]
B_TRANSFORMS = [subtraction_transform, divide_transform]
SLICER = OptimizedSlicer()
PREPROCESSOR = Preprocessor()
AGG_FUNCTIONS = [MaxAgg(), MinAgg(), MeanAgg(), PtpAgg(), TrendAgg(), StdAgg(), SumAgg(), MedianAgg(), VarAgg(), PeakAgg()]


class Extractor:

    def __init__(self,
                 preprocessor = PREPROCESSOR,
                 u_transforms=U_TRANSFORMS,
                 b_transforms=B_TRANSFORMS,
                 slicer=SLICER,
                 agg_functions=AGG_FUNCTIONS,
                 fillna=0):

        self.preprocessor = preprocessor
        self.u_transforms = u_transforms
        self.b_transforms = b_transforms
        self.slicer = slicer
        self.agg_functions = agg_functions
        self.fillna = fillna

        if identity_transform not in self.u_transforms:
            self.u_transforms.append(identity_transform)

    def __calculate_transforms(self, X, y=None):
        print('-- Calculate transforms --')
        X = pd.DataFrame(X)
        self.df = pd.DataFrame()

        for ut in self.u_transforms:
            print('-- Calculate {} transform --'.format(ut.name))
            ut_result = ut.fit_transform(X)
            self.df = pd.concat([self.df, ut_result], axis=1)
            for bt in self.b_transforms:
                print('-- Calculate {} transform --'.format(bt.name))
                bt_result = bt.fit_transform(ut_result)
                self.df = pd.concat([self.df, bt_result], axis=1)

    def __calculate_aggregation_functions(self):
        print('-- Extract features --')
        features = []
        columns = []
        for start, end in self.slicer.intervals:
            print('-- Extract from slice [{} {}) --'.format(str(start), str(end)))
            for column in self.df.columns:
                for agg in self.agg_functions:
                    columns.append(SEPERATOR.join([column, str(start), str(end), agg.name]))
                    features.append(agg.agg(np.stack(self.df[column].values)[:, start:end], axis=1))
        self.feature_vector = pd.DataFrame(np.array(features).T, columns=columns)


    def fit(self, X, y=None, **kwargs):
        self.preprocessor.fit(X)
        self.slicer.fit(X, y)
        self.y_train = y
        return self

    def transform(self, X, y=None, **kwargs):
        X = self.preprocessor.transform(X)
        self.__calculate_transforms(X)
        self.__calculate_aggregation_functions()
        return self.feature_vector.replace(-np.inf, self.fillna).replace(np.inf, self.fillna).fillna(self.fillna)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X, y)
