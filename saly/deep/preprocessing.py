from .. import backend


def preprocess_data(data, train=0.7, validation=0.15, test=0.15):
    """
    Shuffles, log10 transforms and splits the data into train, validation and test sets.
    """

    train_index, validation_index, test_index = backend.get_data_splits(data.shape[0], train, validation, test)
    data = backend.shuffle_data(data, axis=0)
    data.X = backend.normalize_data(data.X)

    train = data[:train_index]
    validation = data[validation_index:test_index]
    test = data[train_index:]

    return train, validation, test
