from collections import OrderedDict


def get_used_markers(genes: list, markers) -> list:
    """
    Returns the marker genes found in the given data set.
    """
    used_markers = []
    for i, gene_name in enumerate(genes):
        used = markers.loc[markers['Name'] == gene_name]
        if not used.empty:
            [used_markers.append(tuple(marker)) for marker in used.values.tolist()]
    return used_markers


def sort_markers_by_type(markers: list) -> OrderedDict:
    """
    Returns an OrderedDictionary of every cell type and its
    corresponding genes found in the given array of markers.
    :param markers: list of used markers
    """
    by_type = OrderedDict()
    for marker in markers:
        gene_name = marker[1]
        c_type = marker[2]

        # check if the key already exists
        if c_type in by_type:
            # check if the gene is already in the array
            if gene_name not in by_type[c_type]:
                by_type[c_type].append(gene_name)

        else:  # if not, create a new entry
            by_type[c_type] = [gene_name]
    return by_type


def get_used_genes(markers_by_type):
    """
    Returns a (unique) array of genes used in the specified array.
    :param markers_by_type
    :return
    """
    used_genes = []
    for c_type in markers_by_type:
        genes = markers_by_type[c_type]
        for gene in genes:
            if gene not in used_genes:
                used_genes.append(gene)
    return used_genes


def get_cell_types(markers: list) -> list:
    """
    Returns an array of cell types given an array of used markers.
    :param markers: used markers
    :return: 2D array of cell_type=>[genes]
    """
    used_types = []
    for marker in markers:
        c_type = marker[2]
        if c_type not in used_types:
            used_types.append(c_type)
    return used_types


def get_rows_to_drop(labels_to_drop: list, labels: list) -> list:
    """
    Returns a list of row indices to drop from a list of row labels.
    :return: list of indices
    """
    return [i for i, c_type in enumerate(labels) if c_type in labels_to_drop]


def get_rows_to_keep(labels_to_drop: list, labels: list) -> list:
    """
    Returns a list of row indices to keep from a list of row labels.
    :return: list of indices
    """
    return [i for i, c_type in enumerate(labels) if c_type not in labels_to_drop]


def check_for_unknown(labels: list, cell_types: list, aliases: dict) -> list:
    """
    Returns cell types not found in either of the provided lists.
    :param labels: list of labels
    :param cell_types: list of cell types
    :param aliases: dictionary of aliases
    :return: unknown labels
    """
    unknown = []
    known_aliases = list(aliases.keys())
    for c_type in labels:
        if c_type not in cell_types and c_type not in known_aliases:
            if c_type not in unknown:
                unknown.append(c_type)

    return unknown
