from numpy.random import permutation
from pandas import DataFrame, Series
from numpy import log10, power


def log_10(data):
    """
    Log10 transformation
    """
    return log10(data + 1)


def from_log10(data):
    """
    Inverse function of the log10 transformation.
    """
    return power(10, data) - 1


def get_data_splits(n: int, train: float, validation: float, test: float) -> (int, int, int):
    """
    Splits the data into train, validation and test sets.
    :param n: length of array to split
    :param train: % of train data
    :param validation: % of validation data
    :param test: % of test data
    :return: Indices of train, validation and test data
    """
    if train + validation + test > 1.0:
        raise ValueError("Data splits sum up to more than 1.0!", train, validation, test)

    train_index = int(n * train)
    validation_index = train_index + int(n * validation)
    test_index = validation_index + int(n * test)

    return train_index, validation_index, test_index


def shuffle_data(data: DataFrame, labels: Series, axis=0) -> (DataFrame, Series):
    """
    Shuffles the given data on the given axis.
    :param data: DataFrame to shuffle
    :param labels: Series to shuffle
    :param axis: row (0) or column (1)
    :return: shuffled DataFrame and Series
    """
    if axis == 0:
        idx = permutation(data.index)
        data = data.reindex(idx)
        labels = labels.reindex(idx)
    elif axis == 1:
        idx = permutation(data.columns.index)
        data = data.reindex(idx, axis=1)
    else:
        raise ValueError("Axis must be 0 or 1; given ", axis)

    return data, labels
