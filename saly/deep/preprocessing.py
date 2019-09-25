from .. import backend
import math


def preprocess_data(data, train=0.7, validation=0.15, test=0.15, splits=None):
    """
    Shuffles, log10 transforms and splits the data into train, validation and test sets.
    """
    if splits is None:
        splits = 1

    train_index, validation_index, test_index = backend.get_data_splits(data.shape[0], train, validation, test)
    data = backend.shuffle_data(data, axis=0)
    
    split = data.shape[0] // splits
    for i in range(splits):
        print("Split", i)
        
        start, finish = i * split, (i + 1) * split
        data.X[start:finish] = backend.normalize_data(data[start:finish])

    train = data[:train_index]
    validation = data[validation_index:test_index]
    test = data[train_index:]

    return train, validation, test
