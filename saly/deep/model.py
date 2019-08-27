from .. import backend
from ..backend import Markers
from keras.layers import Input, Dense, Dropout
from keras.models import Model


def build_model(data, markers, bottleneck_dim=25, intermediate_dim=100, dropout_n=0.1,
                activation='relu', loss='mse', optimizer='adam'):
    """
    Builds the AutoEncoder model.
    :param data: DataFrame to train/test on
    :param markers: list of used markers
    :param bottleneck_dim: number of bottleneck nodes
    :param intermediate_dim: number of dense layer nodes
    :param dropout_n: dropout rate (from 0 to 1.0)
    :param activation: activation function
    :param loss: loss function
    :param optimizer: optimizer function
    :return: AutoEncoder, marker and encoder models
    """
    by_type = backend.sort_markers_by_type(markers)
    input_dim = data.shape[1]
    marker_dim = len(by_type)

    weight_mask = backend.get_weight_mask(by_cell_type=by_type, shape=(marker_dim, input_dim), genes=data.columns)
    # -- Model --
    input_layer = Input(shape=(input_dim,))
    marker_layer = Markers(marker_dim, weight_mask=weight_mask,
                           activation=activation, name='cell_activations')(input_layer)

    dense_in_1 = Dense(intermediate_dim, activation=activation)(marker_layer)
    bottleneck_layer = Dense(bottleneck_dim, activation=activation, name='Bottleneck')(dense_in_1)
    dense_out_1 = Dense(intermediate_dim, activation=activation)(bottleneck_layer)

    dropout = Dropout(rate=dropout_n)(dense_out_1)
    output_layer = Dense(input_dim, activation=activation, name='output')(dropout)
    # --------
    autoencoder_model = Model(input_layer, [marker_layer, output_layer])
    marker_model = Model(input_layer, marker_layer)
    encoder_model = Model(input_layer, bottleneck_layer)

    autoencoder_model.compile(loss={'cell_activations': backend.marker_loss, 'output': loss},
                              loss_weights={'cell_activations': 1., 'output': 100.0},
                              metrics={'cell_activations': backend.marker_prediction_metric},
                              optimizer=optimizer)
    marker_model.compile(loss=loss, optimizer=optimizer)
    encoder_model.compile(loss=loss, optimizer=optimizer)

    return autoencoder_model, marker_model, encoder_model


def build_model_lossless(data, markers, bottleneck_dim=25, intermediate_dim=100, dropout_n=0.1,
                activation='relu', loss='mse', optimizer='adam'):
    """
    Builds the AutoEncoder model.
    :param data: DataFrame to train/test on
    :param markers: list of used markers
    :param bottleneck_dim: number of bottleneck nodes
    :param intermediate_dim: number of dense layer nodes
    :param dropout_n: dropout rate (from 0 to 1.0)
    :param activation: activation function
    :param loss: loss function
    :param optimizer: optimizer function
    :return: AutoEncoder, marker and encoder models
    """
    by_type = backend.sort_markers_by_type(markers)
    input_dim = data.shape[1]
    marker_dim = len(by_type)

    weight_mask = backend.get_weight_mask(by_cell_type=by_type, shape=(marker_dim, input_dim), genes=data.columns)
    # -- Model --
    input_layer = Input(shape=(input_dim,))
    marker_layer = Markers(marker_dim, weight_mask=weight_mask,
                           activation=activation, name='cell_activations')(input_layer)

    dense_in_1 = Dense(intermediate_dim, activation=activation)(marker_layer)
    bottleneck_layer = Dense(bottleneck_dim, activation=activation, name='Bottleneck')(dense_in_1)
    dense_out_1 = Dense(intermediate_dim, activation=activation)(bottleneck_layer)

    dropout = Dropout(rate=dropout_n)(dense_out_1)
    output_layer = Dense(input_dim, activation=activation, name='output')(dropout)
    # --------
    autoencoder_model = Model(input_layer, output_layer)
    marker_model = Model(input_layer, marker_layer)
    encoder_model = Model(input_layer, bottleneck_layer)

    autoencoder_model.compile(loss=loss, optimizer=optimizer)
    marker_model.compile(loss=loss, optimizer=optimizer)
    encoder_model.compile(loss=loss, optimizer=optimizer)

    return autoencoder_model, marker_model, encoder_model


def train_model(model, data, labels, markers, marker_aliases, epochs,
                validation_data=None, batch_size=256, verbose=1, callbacks=None):
    """
    Trains the Keras model.
    :return: a Keras train history object
    """
    if validation_data is not None:
        validation_labels = validation_data[1]
        val_labels_one_hot = backend.one_hot_encode(validation_labels, markers, marker_aliases)
        validation_data = (validation_data[0], {'cell_activations': val_labels_one_hot, 'output': validation_data[0]})

    labels_one_hot = backend.one_hot_encode(labels, markers, marker_aliases)
    history = model.fit(data, {'cell_activations': labels_one_hot, 'output': data},
                        epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose)

    return history


def test_model(model, data_x, data_y, markers, aliases, verbose=0):
    """
    Evaluates the model on the given data
    :return: the data's loss score
    """
    labels_one_hot = backend.one_hot_encode(data_y, markers, aliases)
    results = model.evaluate(data_x, {'cell_activations': labels_one_hot, 'output': data_x}, verbose=verbose)

    print("Test reconstruction loss:", round(results[2], 8))
    print("Test prediction accuracy:", round(results[3] * 100, 3), "%")

    return results
