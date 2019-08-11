from collections import OrderedDict
from Orange.data import Table


def get_markers_by_species(species: str, markers: Table) -> list:
    """
    Returns an array of marker genes of the specified species.
    :param species: "Mouse", "Human" etc.
    :param markers: Array of marker genes
    :return:
    """
    markers_by_species = []
    for meta in markers.metas:
        if meta[0] == species:
            markers_by_species.append(meta)
    return markers_by_species


def get_used_markers(genes, markers):
    """
    Returns the marker genes found in the given data set.
    :param genes:
    :param markers:
    :return:
    """
    used_markers = []
    for i, gene_name in enumerate(genes):
        for marker in markers:
            if gene_name == marker[1]:
                used_markers.append(tuple(marker.tolist()))
    return used_markers


def sort_markers_by_type(markers):
    """
    Returns an OrderedDictionary of every cell type and its
    corresponding genes found in the given array of markers.
    :param markers:
    :return:
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


def get_cell_types(markers):
    """
    Returns an array of cell types given an array of markers sorted by cell type.
    :param markers:
    :return:
    """
    used_types = []
    for marker in markers:
        c_type = marker[2]
        used_types.append(c_type)
    return used_types


def get_rows_to_drop(labels_to_drop, labels):
    """
    Returns a list of row indices to drop from a list of row labels.
    :param labels_to_drop:
    :param labels:
    :return:
    """
    to_drop = []
    for i, c_type in enumerate(labels):
        if c_type in labels_to_drop:
            to_drop.append(i)
    return to_drop


def check_for_unknown(labels, cell_types, aliases):
    """
    Returns cell types not found in either of the provided lists.
    :param labels:
    :param cell_types:
    :param aliases:
    :return:
    """
    unknown = []
    known_aliases = list(aliases.keys())
    for c_type in labels:
        if c_type not in cell_types and c_type not in known_aliases:
            if c_type not in unknown:
                unknown.append(c_type)

    return unknown
