import numpy as np
from scipy import sparse
import scanpy as scpy


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
    

def var(x, axis=None, ddof=0):
    """ Equivalent of np.var that supports sparse and dense matrices. """
    if not sparse.issparse(x):
        return np.var(x, axis, ddof=ddof)

    result = x.multiply(x).mean(axis) - np.square(x.mean(axis))
    result = np.squeeze(np.asarray(result))

    # Apply correction for degrees of freedom
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    result *= n / (n - ddof)

    return result


def std(x, axis=None, ddof=0):
    """ Equivalent of np.std that supports sparse and dense matrices. """
    return np.sqrt(var(x, axis=axis, ddof=ddof))


def normalize_data(data):
    
    scpy.pp.normalize_total(data, target_sum=1e6)
    scpy.pp.log1p(data)
    
    c_std = std(data.X, axis=0)
    c_mean = data.X.mean(axis=0).A[0]
    
    arr = data.X.toarray()
    arr = np.apply_along_axis(standardize_data, 1, arr, c_mean, c_std)
    
    data.X = sparse.csr_matrix(arr)
    return data


def standardize_data(data, mean, std):
    sub = np.subtract(data, mean)
    return np.divide(sub, std, out=np.zeros_like(sub), where=std!=0)
