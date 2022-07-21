import os
from sktime.datasets import load_from_tsfile
from effects.utils.utils import rename_dataFrame_columns

BASE_DATASETS_PATH = r'./datasets/UAE/'

def get_all_datasets():
    return os.listdir(BASE_DATASETS_PATH)
	
def load(dataset):
    X_train, y_train = load_from_tsfile(BASE_DATASETS_PATH+'/'+dataset+'/'+dataset+'_TRAIN.ts')
    X_test, y_test = load_from_tsfile(BASE_DATASETS_PATH+'/'+dataset+'/'+dataset+'_TEST.ts')
    for col in X_train.columns:
        X_train[col] = X_train[col].apply(lambda x: x.values)
        X_test[col] = X_test[col].apply(lambda x: x.values)
		
    X_train = rename_dataFrame_columns(X_train)
    X_test = rename_dataFrame_columns(X_test)
    return X_train, X_test, y_train, y_test