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


def shuffle_data(data: DataFrame, labels: Series) -> (DataFrame, Series):
    """
    Shuffles the given data.
    :param data: DataFrame to shuffle
    :param labels: Series to shuffle
    :return: shuffled DataFrame and Series
    """
    idx = permutation(data.index)
    data = data.reindex(idx)
    labels = labels.reindex(idx)

    return data, labels
