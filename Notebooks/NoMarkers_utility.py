import saly.backend as backend

from keras.layers import Input, Dense, Dropout
from keras.models import Model

from saly.backend import Markers

import matplotlib.pyplot as plt


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

    dense_in_1 = Dense(intermediate_dim, activation=activation)(input_layer)
    bottleneck_layer = Dense(bottleneck_dim, activation=activation, name='Bottleneck')(dense_in_1)
    dense_out_1 = Dense(intermediate_dim, activation=activation)(bottleneck_layer)

    dropout = Dropout(rate=dropout_n)(dense_out_1)
    output_layer = Dense(input_dim, activation=activation, name='output')(dropout)
    # --------
    autoencoder_model = Model(input_layer, output_layer)
    bottleneck_model = Model(input_layer, bottleneck_layer)

    autoencoder_model.compile(loss=loss, optimizer=optimizer)
    bottleneck_model.compile(loss=loss, optimizer=optimizer)

    return autoencoder_model, bottleneck_model


def train_model(model, data, epochs,
                validation_data=None, batch_size=256, verbose=1, callbacks=None):
    """
    Trains the Keras model.
    :return: a Keras train history object
    """

    history = model.fit(data, data,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose)

    return history


def test_model(model, data_x, verbose=0):
    """
    Evaluates the model on the given data
    :return: the data's loss score
    """
    results = model.evaluate(data_x, data_x, verbose=verbose)

    return results


def plot_model_history(history):
    """
    Draws a model's training history.
    :param history: a Keras history object
    """
    fig, ax1 = plt.subplots(1, 1, sharex='all', figsize=(7, 7), dpi=80)
    output_loss = history.history['loss']

    epochs = range(1, len(output_loss) + 1)

    ax1.plot(epochs, output_loss, 'b--', label='Training loss')

    if 'val_loss' in history.history.keys():
        val_output_loss = history.history['val_loss']

        ax1.plot(epochs, val_output_loss, 'r--', label='Validation reconstruction loss')
        ax1.set_title('Training and validation loss')

    else:
        ax1.set_title('Training loss per epoch')

    fig.text(0.04, 0.5, 'Loss', va='center',
             rotation='vertical')
    plt.xlabel('Epochs')
    ax1.legend()

    plt.show()
