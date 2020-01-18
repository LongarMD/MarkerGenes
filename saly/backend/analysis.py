import pickle
from random import uniform, seed
from time import time
from collections import OrderedDict
from saly import backend
import pandas as pd
from sklearn import metrics
import numpy as np


def get_top_activated_indices(n: int, cell_type_activations: list) -> list:
    """
    Returns a list of indices of top n activations
    :param n: number of top activations to return
    :param cell_type_activations: list of Marker Layer node activations
    :return: list of n indices
    """
    tops = []
    for cell in cell_type_activations:
        top = cell.argsort()[-n:][::-1]
        tops.append(top)

    return tops


def index_to_cell_type(indices, cell_types):
    """
    Converts Marker Layer activation indices to cell types
    :param indices: list of Marker Layer activation indices
    :param cell_types: list of cell types in the marker layer
    :return: list of activated cell types
    """
    activated_types = []

    for cell in indices:
        activated_types.append([cell_types[i] for i in cell])

    return activated_types


def get_random_colour():
    """
    Generates random RGB values.
    """
    seed(time())
    pastel_factor = uniform(0.0, 0.8)
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [uniform(0, 1.0) for _ in [1, 2, 3]]]


def save_label_colours(colours: dict, path: str) -> str:
    """
    Saves a dictionary of label : colour into a pickle file
    :param colours: a dictionary of label : colour pairs
    :param path: path to save folder
    :return: file location
    """
    name = path + 'label_colours.pickle'

    pickle_out = open(name, "wb")
    pickle.dump(colours, pickle_out)
    pickle_out.close()

    return name


def load_label_colours(path: str) -> dict:
    """
    Loads a dictionary from a pickle file
    :param path: path to file
    :return: loaded dictionary
    """
    pickle_in = open(path, "rb")
    colours = pickle.load(pickle_in)

    return colours


def get_graph_labels(handles, plt_labels):
    by_label = OrderedDict(zip(plt_labels, handles))
    by_label = OrderedDict(sorted(by_label.items()))

    return by_label.values(), by_label.keys()


def get_class_evaluation(labels, cell_activations, markers, aliases):
    cell_types = backend.get_cell_types(markers)
    top_activations = backend.get_top_activated_indices(1, cell_activations)

    predictions = backend.index_to_cell_type(top_activations, cell_types)
    predictions = pd.Series([p[0] for p in predictions])

    y_true = labels
    y_true = y_true.apply(lambda x: aliases[x] if x in aliases.keys() else x)
    unique_classes = sorted(y_true.append(predictions).unique())
    true_classes = sorted(y_true.unique())

    cm = metrics.confusion_matrix(y_true, predictions)

    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (recall * precision) / (recall + precision)

    results = {}
    for i, cls in enumerate(unique_classes):
        if cls in true_classes:
            result = {'tp': tp[i], 'tn': tn[i], 'fp': fp[i], 'fn': fn[i],
                      'precision': precision[i], 'recall': recall[i], 'f1': f1_score[i]}
            results.update({cls: result})
    return results
