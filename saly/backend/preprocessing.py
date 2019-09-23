import numpy as np
from scipy import sparse


def log_10(data):
    """
    Log10 transformation
    """
    return np.log10(data + 1)


def from_log10(data):
    """
    Inverse function of the log10 transformation.
    """
    return np.power(10, data) - 1


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


def shuffle_data(data, axis=0):
    """
    Shuffles the data on the given axis.
    """
    if axis == 0:
        idx = np.random.permutation(data.obs['labels'].index)
        data = data[idx, :]
    elif axis == 1:
        idx = np.random.permutation(data.var_names.index)
        data = data[:, idx]
    else:
        raise ValueError("Axis must be 0 or 1; given ", axis)

    return data


def normalize_data(data):
    """
    Normalizes gene activations
    """
    n_columns = data.shape[1]

    normalized = sparse.lil_matrix(data.shape)
    for i in range(n_columns):
        column = data[:, i]
        max_n = column[column.argmax(), 0]

        if max_n != 0.:
            column = column / max_n
        normalized[:, i] = column

    return normalized.tocsc()
