import pandas as pd
import numpy as np
import effects.config as config

class Extractor:

    def __init__(self,
                 preprocessor=config.PREPROCESSOR,
                 u_transforms=config.U_TRANSFORMS,
                 b_transforms=config.B_TRANSFORMS,
                 slicer=config.SLICER,
                 agg_functions=config.AGG_FUNCTIONS,
                 fillna=0):

        self.y = None
        self.preprocessor = preprocessor
        self.u_transforms = u_transforms
        self.b_transforms = b_transforms
        self.slicer = slicer
        self.agg_functions = agg_functions
        self.fillna = fillna
        self.slices = []
        self.feature_vector = None

    def fit(self, X, y=None, **kwargs):
        self.preprocessor.fit(X)
        self.slices = self.slicer.get_slices(X, y)
        self.y = y
        return self

    def transform(self, X, y=None, verbose=1, **kwargs):

        if verbose == 1:
            def verboseprint(s):
                print(s)
        else:
            verboseprint = lambda s: None

        self.verboseprint = verboseprint

        X = self.preprocessor.transform(X)

        self.transformed_data = pd.DataFrame()
        self.feature_vector = pd.DataFrame()

        self.__calculate_transforms(X)
        self.__calculate_aggregation_functions()

        self.feature_vector = self.feature_vector.replace(-np.inf, self.fillna).replace(np.inf, self.fillna).fillna(self.fillna)

        return self.feature_vector

    def fit_transform(self, X, y=None, verbose=1, **kwargs):
        return self.fit(X, y).transform(X, verbose=verbose)

    def __calculate_transforms(self, X, y=None):
        self.verboseprint('-- Calculate transforms --')
        columns_pairs = []
        for ut in self.u_transforms:
            ut_columns = []
            self.verboseprint('-- Calculate {} transform --'.format(ut.name))
            for col in X.columns:
                col_name = config.SEPARATOR.join([col, ut.name])
                self.transformed_data[col_name] = ut.transform(X[col])
                ut_columns.append(col_name)
            columns_pairs += [(a, b) for idx, a in enumerate(ut_columns) for b in ut_columns[idx + 1:]]
        for bt in self.b_transforms:
            self.verboseprint('-- Calculate {} transform --'.format(bt.name))
            for col1, col2 in columns_pairs:
                self.transformed_data[config.SEPARATOR.join([col1, col2, bt.name])] = bt.transform(
                    self.transformed_data[[col1, col2]])

    def __calculate_aggregation_functions(self):
        self.verboseprint('-- Extract features --')
        features = []
        for start, end in self.slices:
            self.verboseprint('-- Extract from slice [{} {}) --'.format(str(start), str(end)))
            for column in self.transformed_data.columns:
                for agg in self.agg_functions:
                    features.append(
                        pd.DataFrame(
                            agg.transform(np.stack(self.transformed_data[column].values)[:, start:end]),
                            columns=[config.SEPARATOR.join([column, str(start), str(end), agg.name])]
                        )
                    )
        self.feature_vector = pd.concat(features, axis=1)


    def explain(self, feature_vector=None):

        assert feature_vector is not None or self.feature_vector is not None

        if feature_vector is None:
            feature_vector = self.feature_vector