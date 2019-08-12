from .. import backend
from numpy import mean, transpose
import matplotlib.pyplot as plt
from seaborn import distplot


def get_results(labels: list, cell_activations: list, markers: list, aliases: dict) -> None:
    """
    Compares the real labels to the cell type activations.
    :param labels: Data labels
    :param cell_activations: Activations of the Marker Layer nodes
    :param markers: list of used markers
    :param aliases: dict of `label : name_in_marker_db` aliases
    """
    cell_types = backend.get_cell_types(markers)

    top_activations = backend.get_top_activations(1, cell_activations)
    predictions = backend.index_to_cell_type(top_activations, cell_types)

    correct = 0
    n = len(cell_activations)
    for i, prediction in enumerate(predictions):
        label = labels[i]
        prediction = prediction[0]  # because predictions is a 2D array

        if prediction == label:
            correct += 1

        elif label in aliases.keys():
            if aliases[label] == prediction:
                correct += 1

    print("Correct predictions: {c} out of {n} ({p}%)".format(c=correct, n=n, p=round(100 * (correct / n), 2)))


def plot_model_history(history):
    """
    Draws a model's training history.
    :param history: a Keras history object
    """

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')

    if 'val_loss' in history.history.keys():
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'b', label='Validation loss', c='orange')
        plt.title('Training and validation loss per epoch')
    else:
        plt.title('Training loss per epoch')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_activation_distribution(cell_activations, markers):
    """
    Plots the average cell type activation.
    :param cell_activations: Marker layer node activations
    :param markers: list of used markers
    """
    n_cell_types = len(backend.get_cell_types(markers))
    sorted_indices = backend.get_top_activations(n_cell_types, cell_activations)

    sorted_activations = []
    for i, cell in enumerate(cell_activations):
        sorted_activations.append([cell[j] for j in sorted_indices[i]])

    average = [mean(column) for column in transpose(sorted_activations)]

    distplot(average)
