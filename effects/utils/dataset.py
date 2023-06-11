from sktime.datasets import load_from_tsfile
import os

def ts_format_data_loader(path):
    """
    :param path: Int
    :return: ts data Separate to train/test in DataFrame format
    """
    dataset_name = os.path.basename(path.strip("/"))
    X_train, y_train = load_from_tsfile(os.path.join(path, dataset_name+'_TRAIN.ts'))
    X_test, y_test = load_from_tsfile(os.path.join(path, dataset_name+'_TEST.ts'))

    for col in X_train.columns:
        X_train[col] = X_train[col].apply(lambda x: x.values)
        X_test[col] = X_test[col].apply(lambda x: x.values)

    return X_train, X_test, y_train, y_test