from collections import OrderedDict

import matplotlib.pyplot as plt
from numpy import mean, transpose
from seaborn import distplot

from .. import backend


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


def plot_activation_distribution(cell_activations, markers, title=''):
    """
    Plots the average cell type activation.
    :param cell_activations: Marker layer node activations
    :param markers: list of used markers
    :param title: graph title
    """
    n_cell_types = len(backend.get_cell_types(markers))
    sorted_indices = backend.get_top_activations(n_cell_types, cell_activations)

    sorted_activations = []
    for i, cell in enumerate(cell_activations):
        sorted_activations.append([cell[j] for j in sorted_indices[i]])

    average = [mean(column) for column in transpose(sorted_activations)]

    distplot(average).set_title(title)


def get_label_colours(labels):
    """
    Generates a dictionary of colours for each label
    :param labels: list of labels
    :return: dictionary of label : colour
    """
    colours = {}
    for label in labels.unique():
        colour = (backend.get_random_colour())
        colours.update({label: colour})

    return colours


def plot_label_colours(colours):
    """
    Plots a list of colours
    :param colours:
    """
    if type(colours) == dict:
        colours = colours.values()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = list(range(0, len(colours) * 2, 2))
    y = [0] * len(x)
    c = list(colours)

    ax.scatter(x, y, s=400, c=c)
    ax.set_title('Cell type colours')


def draw_embedding(x, y, model, colours=None, alpha=1.0, graph_title=''):
    """
    Draws a scatter plot of the provided embedding model
    :param x: Data x
    :param y: Data labels
    :param model: embedder e.g. tSNE or PCA
    :param colours: label colours
    :param alpha: Node alpha
    :param graph_title: Graph title
    """
    if colours is None:
        colours = get_label_colours(y)

    model_out = model.fit_transform(x)

    plt.clf()
    plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

    for i, entry in enumerate(model_out):
        plt.scatter(entry[0], entry[1], color=colours[y.iloc[i]], label=y.iloc[i], alpha=alpha, edgecolors='black')

    # Get legend
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(plt_labels, handles))
    by_label = OrderedDict(sorted(by_label.items()))

    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0., 1, 0, .1))

    plt.title(graph_title)
    plt.axis('off')
    plt.show()


def draw_embeddings(x, y, model, colours=None, graph_title=''):
    if colours is None:
        colours = get_label_colours(y)

    model_out = model.fit_transform(x)

    plt.clf()
    plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

    for i, entry in enumerate(model_out):
        plt.scatter(entry[0], entry[1], color=colours[y.iloc[i]], label=y.iloc[i])

    # Get legend
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(plt_labels, handles))
    by_label = OrderedDict(sorted(by_label.items()))

    plt.legend(by_label.values(), by_label.keys())#, bbox_to_anchor=(0., 1, 0, .1))

    plt.title(graph_title)
    plt.axis('off')
    plt.show()
