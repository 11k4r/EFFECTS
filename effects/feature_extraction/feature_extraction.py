from effects.feature_extraction.conf import *
from effects.utils.utils import generate_name
import pandas as pd
import numpy as np

def extract_features(df):
    base_columns = df.columns
    for column in base_columns:
        for uni_transform in UNIVARIATE_TRANSFORMS:
            df[generate_name([column, uni_transform.__name__])] = df[column].apply(lambda x: uni_transform(x))
			
    pair_of_columns = [(a, b) for idx, a in enumerate(df.columns) for b in df.columns[idx + 1:] if 
         ('_' not in a and '_' not in b) or (a.split('_')[-1] == b.split('_')[-1])]
    for pair in pair_of_columns:
        column1, column2 = pair
        for bi_transform in BIVARIATE_TRANSFORMS:
            df[generate_name([column1, column2, bi_transform.__name__])] = df.apply(lambda x: bi_transform(x[column1], x[column2]), axis=1)
			
    vec = pd.DataFrame()
	
    vec = vec.replace(np.inf, np.nan).replace(-np.inf, np.nan).fillna(0).astype(float)
    vec[vec > M] = M
    vec[vec < -M] = -M	
	
    return extract_global_features(df)
	
def extract_global_features(df):
    global_features = pd.DataFrame()
    for column in df.columns:
        for agg_func in AGGREGATION_FUNCTIONS:
            global_features[generate_name([column, agg_func.__name__])] = df[column].apply(lambda x: agg_func(np.nan_to_num(x, 0)))
    return global_features

