import saly.backend as backend

from keras.layers import Input, Dense
from keras.models import Model

from saly.backend import Markers

import numpy as np
import matplotlib.pyplot as plt


def build_model(data, markers, optimizer='adam'):
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
                           activation='softmax', name='cell_activations')(input_layer)
    # --------
    marker_model = Model(input_layer, marker_layer)
    marker_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                         metrics=['accuracy'])


    return marker_model


def build_dense(data, markers, optimizer='adam'):
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

    weight_mask = np.ones(shape=(marker_dim, input_dim))
    # -- Model --
    input_layer = Input(shape=(input_dim,))
    marker_layer = Markers(marker_dim, weight_mask=weight_mask,
                           activation='softmax', name='cell_activations')(input_layer)
    # --------
    marker_model = Model(input_layer, marker_layer)
    marker_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                         metrics=['accuracy'])


    return marker_model


def train_model(model, data, labels, markers, marker_aliases, epochs,
                validation_data=None, batch_size=256, verbose=1, callbacks=None):
    """
    Trains the Keras model.
    :return: a Keras train history object
    """
    if validation_data is not None:
        validation_labels = validation_data[1]
        val_labels_one_hot = backend.one_hot_encode(validation_labels, markers, marker_aliases)
        validation_data = (validation_data[0], val_labels_one_hot)

    labels_one_hot = backend.one_hot_encode(labels, markers, marker_aliases)
    history = model.fit(data, labels_one_hot,
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
    results = model.evaluate(data_x, labels_one_hot, verbose=verbose)

    print("Test prediction loss:", round(results[0], 8))
    print("Test prediction accuracy:", round(results[1] * 100, 3), "%")

    return results


def plot_model_history(history):
    """
    Draws a model's training history.
    :param history: a Keras history object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(7, 7), dpi=80)
    output_loss = history.history['loss']

    accuracy = history.history['acc']
    accuracy = [i * 100 for i in accuracy]

    epochs = range(1, len(output_loss) + 1)

    ax1.plot(epochs, output_loss, 'b--', label='Training loss')
    ax2.plot(epochs, accuracy, 'g--', label='Training accuracy')

    if 'val_loss' in history.history.keys():
        val_output_loss = history.history['val_loss']

        val_accuracy = history.history['val_acc']
        val_accuracy = [i * 100 for i in val_accuracy]

        ax1.plot(epochs, val_output_loss, 'r--', label='Validation prediction loss')
        ax1.set_title('Training and validation loss')

        ax2.plot(epochs, val_accuracy, 'g-', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
    else:
        ax1.set_title('Training loss per epoch')
        ax2.set_title('Training accuracy per epoch')

    fig.text(0.04, 0.5, 'Accuracy (%)                                                    Loss', va='center',
             rotation='vertical')
    plt.xlabel('Epochs')
    ax1.legend()
    ax2.legend()

    plt.show()
