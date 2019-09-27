from .. import backend
import pandas as pd
import numpy as np


def preprocess_data(data, train=0.7, validation=0.15, test=0.15, splits=None):
    """
    Shuffles, log10 transforms and splits the data into train, validation and test sets.
    """
    if splits is None:
        splits = 1

    train_index, validation_index, test_index = backend.get_data_splits(data.shape[0], train, validation, test)
    data = backend.shuffle_data(data, axis=0)
    
    split_size = data.shape[0] // splits
    data = backend.normalize_data(data.copy(), split_size)

    train = data[:train_index]
    validation = data[validation_index:test_index]
    test = data[train_index:]

    return train, validation, test


def mark_as_unlabelled(data):
    
    if 'labels' in data.obs_keys():
        data.obs['labels'] = pd.Series(np.repeat(-1, data.shape[0]),
                                       index=data.obs['labels'].index)
    else:
        data.obs.insert(0, 'labels', np.repeat(-1, data.shape[0]), True)
    return data
