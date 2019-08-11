from .. import backend


def preprocess_data(data, labels, train=0.7, validation=0.15, test=0.15):
    """
    Shuffles, log10 transforms and splits the data into train, validation and test sets.
    :param data: data DataFrame
    :param labels: Series of labels
    :param train: % of train data
    :param validation: % of validation data
    :param test: % of test data
    :return: validation and test sets.
    """
    train_index, validation_index, test_index = backend.get_data_splits(data.shape[0], train, validation, test)
    data, labels = backend.shuffle_data(data, labels)
    data = backend.log_10(data)

    train_x = data[:train_index]
    train_y = labels[:train_index]

    validation_x = data[validation_index:test_index]
    validation_y = labels[validation_index:test_index]

    test_x = data[train_index:]
    test_y = labels[train_index:]

    return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)
