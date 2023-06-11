from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
#from itertools import cycle

from effects.config import SEPARATOR

class Explorer:

    def __init__(self, extractor, y=None, identity='Identity'):
        self.extractor = extractor
        if y is not None:
            self.y = y
        else:
            self.y = extractor.y
        self.identity = identity
        self.classes_ = np.unique(self.y)
        self.num_classes = len(self.classes_)
        self.feature_scores = None
        self.length = len(extractor.transformed_data.iloc[0][0])
        self.feature_max_score = None
        self.calculate_feature_scores()

        self.slices_score = self.feature_scores.groupby('slice_id')['score'].sum().to_dict()
        self.slice_max_score = self.feature_scores.groupby('slice_id').size()[0] * self.feature_max_score

    def wasserstein_score(self, a, b):
        scaler = MinMaxScaler().fit(np.concatenate([a, b]).reshape(-1, 1))
        return wasserstein_distance(scaler.transform(a.reshape(-1, 1)).reshape(-1),
                                    scaler.transform(b.reshape(-1, 1)).reshape(-1))

    def calculate_feature_scores(self):
        self.feature_scores = pd.DataFrame(self.extractor.feature_vector.columns, columns=['name'])
        self.feature_scores['temp'] = self.feature_scores['name'].apply(lambda x: x.split(SEPARATOR))
        self.feature_scores['bi_transform'] = self.feature_scores['temp'].apply(
            lambda x: x[4] if len(x) == 8 else np.nan)
        self.feature_scores['dim_1'] = self.feature_scores['temp'].apply(lambda x: x[0])
        self.feature_scores['dim_1_uni_transform'] = self.feature_scores['temp'].apply(lambda x: x[1])
        self.feature_scores['dim_2'] = self.feature_scores['temp'].apply(lambda x: x[2] if len(x) == 8 else np.nan)
        self.feature_scores['dim_2_uni_transform'] = self.feature_scores['temp'].apply(
            lambda x: x[3] if len(x) == 8 else np.nan)
        self.feature_scores['slice_start'] = self.feature_scores['temp'].apply(lambda x: x[-3])
        self.feature_scores['slice_end'] = self.feature_scores['temp'].apply(lambda x: x[-2])
        self.feature_scores['slice_id'] = self.feature_scores['slice_start'].map(str) + '_' + self.feature_scores[
            'slice_end'].map(str)
        self.feature_scores['agg'] = self.feature_scores['temp'].apply(lambda x: x[-1])
        self.feature_scores.drop('temp', axis=1, inplace=True)
        self.feature_scores.replace(self.identity, 'None', inplace=True)

        classes_pairs = [(a, b) for idx, a in enumerate(self.classes_)
                         for b in self.classes_[idx + 1:]]

        distances = {}
        for cls1, cls2 in classes_pairs:
            distances[SEPARATOR.join([cls1, cls2])] = []

        for col in tqdm(self.feature_scores['name']):
            for cls1, cls2 in classes_pairs:
                a = self.extractor.feature_vector[self.y == cls1][col].values
                b = self.extractor.feature_vector[self.y == cls2][col].values
                pair_score = self.wasserstein_score(a, b)
                distances[SEPARATOR.join([cls1, cls2])].append(pair_score)

        for cls1, cls2 in classes_pairs:
            self.feature_scores[SEPARATOR.join([cls1, cls2])] = np.array(distances[SEPARATOR.join([cls1, cls2])])
        self.feature_scores['score'] = self.feature_scores[self.feature_scores.columns[-len(classes_pairs):]].sum(
            axis=1)
        self.feature_max_score = len(classes_pairs)

        for y in self.classes_:
            columns = [c for c in self.feature_scores.columns if SEPARATOR in c and y in c]
            self.feature_scores[SEPARATOR.join([y, 'score'])] = self.feature_scores[columns].sum(axis=1) / len(columns)

        self.feature_scores = self.feature_scores.sort_values('score', ascending=False)