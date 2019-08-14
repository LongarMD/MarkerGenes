from .. import backend
from ..backend import Markers
from numpy import save, load
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.backend import manual_variable_initialization


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
    marker_layer = Markers(marker_dim, weight_mask=weight_mask, activation=activation)(input_layer)

    dense_in_1 = Dense(intermediate_dim, activation=activation)(marker_layer)
    bottleneck_layer = Dense(bottleneck_dim, activation=activation, name='Bottleneck')(dense_in_1)
    dense_out_1 = Dense(intermediate_dim, activation=activation)(bottleneck_layer)

    dropout = Dropout(rate=dropout_n)(dense_out_1)
    output_layer = Dense(input_dim, activation=activation, name='Output')(dropout)
    # --------
    autoencoder_model = Model(input_layer, output_layer)
    marker_model = Model(input_layer, marker_layer)
    encoder_model = Model(input_layer, bottleneck_layer)

    autoencoder_model.compile(loss=loss, optimizer=optimizer)
    marker_model.compile(loss=loss, optimizer=optimizer)
    encoder_model.compile(loss=loss, optimizer=optimizer)

    return autoencoder_model, marker_model, encoder_model


def train_model(model, data, epochs, validation_data=None, batch_size=256, verbose=1, callbacks=None):
    """
    Trains the Keras model.
    :return: a Keras train history object
    """
    if validation_data is not None:
        validation_data = (validation_data, validation_data)

    history = model.fit(data, data,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose)

    return history


def test_model(model, test_data, verbose=0):
    """
    Evaluates the model on the given data
    :return: the data's loss score
    """
    loss = model.evaluate(test_data, test_data, verbose=verbose)
    return loss
