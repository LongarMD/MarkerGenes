from .. import backend
from ..backend import Partial
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import initializers as inits

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from anndata import AnnData

from .data import drop_unused_genes


def build_model(data, markers, bottleneck_dim=25, intermediate_dim=100, dropout_n=0.1,
                activation='relu', loss='mse', optimizer='adam', supervised=False):
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
    :param supervised: should the model use a categorical loss on the marker layer?
    :return: AutoEncoder, marker and encoder models
    """
    by_type = backend.sort_markers_by_type(markers)
    input_dim = data.shape[1]
    marker_dim = len(by_type)
    partial_dim = backend.get_partially_dense_size(by_type)

    partially_dense_mask = backend.get_partially_dense_mask(by_cell_type=by_type, genes=data.var_names)
    weight_mask = backend.get_marker_mask(by_cell_type=by_type)

    # -- Model --
    input_layer = Input(shape=(input_dim,))

    partial_input = Partial(partial_dim, weight_mask=partially_dense_mask, use_bias=True,
                            kernel_initializer=inits.ones(),
                            activation=activation)(input_layer)
    
    marker_layer = Partial(marker_dim, weight_mask=weight_mask, use_bias=True,
                           kernel_initializer=inits.ones(),
                           activation=activation, name='cell_activations')(partial_input)

    dense_in_1 = Dense(intermediate_dim, activation=activation)(marker_layer)
    bottleneck_layer = Dense(bottleneck_dim, activation=activation, name='Bottleneck')(dense_in_1)
    dense_out_1 = Dense(intermediate_dim, activation=activation)(bottleneck_layer)

    dropout = Dropout(rate=dropout_n)(dense_out_1)
    output_layer = Dense(input_dim, activation=activation, name='output')(dropout)
    # --------
    autoencoder_model = Model(input_layer, [marker_layer, output_layer])
    marker_model = Model(input_layer, marker_layer)
    encoder_model = Model(input_layer, bottleneck_layer)

    if supervised:
        autoencoder_model.compile(loss={'cell_activations': backend.marker_loss, 'output': loss},
                                  loss_weights={'cell_activations': 1., 'output': 1.},
                                  metrics={'cell_activations': backend.marker_prediction_metric},
                                  optimizer=optimizer)
    else:
        autoencoder_model.compile(loss={'cell_activations': backend.null_loss, 'output': loss},
                                  loss_weights={'cell_activations': 1., 'output': 1000.},
                                  metrics={'cell_activations': backend.marker_prediction_metric},
                                  optimizer=optimizer)
    marker_model.compile(loss=loss, optimizer=optimizer)
    encoder_model.compile(loss=loss, optimizer=optimizer)

    return autoencoder_model, marker_model, encoder_model


def train_model(model, data, markers, marker_aliases, epochs,
                validation_data=None, batch_size=256, verbose=1, callbacks=None):
    """
    Trains the Keras model.
    :return: a Keras train history object
    """
    if validation_data is not None:
        validation_labels = validation_data.obs['labels']
        val_labels_one_hot = backend.one_hot_encode(validation_labels, markers, marker_aliases)
        validation_data = (validation_data.X, {'cell_activations': val_labels_one_hot, 'output': validation_data.X})

    labels_one_hot = backend.one_hot_encode(data.obs['labels'], markers, marker_aliases)
    history = model.fit(data.X, {'cell_activations': labels_one_hot, 'output': data.X},
                        epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose)

    return history


def test_model(model, data, markers, aliases, verbose=0):
    """
    Evaluates the model on the given data
    :return: the data's loss score
    """
    labels_one_hot = backend.one_hot_encode(data.obs['labels'], markers, aliases)
    results = model.evaluate(data.X, {'cell_activations': labels_one_hot, 'output': data.X}, verbose=verbose)

    print("Test reconstruction loss:", round(results[2], 8))
    print("Test prediction accuracy:", round(results[3] * 100, 3), "%")

    return results


def get_baseline(data, markers):
    by_type = backend.sort_markers_by_type(markers)
    matrix = lil_matrix((data.shape[0], len(by_type)))

    for i, c_type in enumerate(by_type):
        column_matrix = len(data[:, by_type[c_type]].X.shape) > 1

        if column_matrix:
            result = data[:, by_type[c_type]].X.sum(axis=1)
            result = result / len(by_type[c_type])
            matrix[:, i] = np.reshape(result, matrix[:, i].shape)
        else:
            matrix[:, i] = np.zeros(matrix[:, i].shape)

    var = pd.DataFrame(index=[c_type for c_type in by_type])
    return AnnData(X=matrix.tocsc(), obs=data.obs, var=var)
