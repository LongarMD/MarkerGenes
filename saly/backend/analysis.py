

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

