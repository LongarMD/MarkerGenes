import pickle
from random import uniform, seed
from time import time
from collections import OrderedDict


def get_top_activations(n: int, cell_type_activations: list) -> list:
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
