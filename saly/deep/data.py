from .. import backend
import anndata as ann
import numpy as np
import pandas as pd


def load_h5ad(path: str):
    """
    Loads a data set from the h5ad format as an AnnData object
    """
    data_set = ann.read_h5ad(path)
    return data_set


def save_h5ad(filename: str):
    ann.write(filename)


def load_markers(marker_path: str, species: str) -> pd.DataFrame:
    """
    Loads marker genes of the specified species.
    :param marker_path: The path to the tab separated file
    :param species: "Mouse", "Human" etc.
    :return: List of markers
    """

    markers_db = pd.read_csv(marker_path, delimiter='\t')
    markers = markers_db[markers_db['Organism'] == species]

    return markers


def get_mutual_markers(data_sets: list, markers_db: list) -> list:
    """
    :param data_sets: list of data sets
    :param markers_db: a list of markers
    :return: an intersection of markers used in the provided data set(s).
    """
    gene_sets = [data_set.var_names for data_set in data_sets]

    mutual = set(backend.get_used_markers(genes=gene_sets[0], markers=markers_db))
    for genes in gene_sets[1:]:
        mutual.intersection_update(backend.get_used_markers(genes=genes, markers=markers_db))

    return list(mutual)


def drop_cell_types(n, markers):
    """
    Removes every cell type with less <= n used marker genes
    """
    by_type = backend.sort_markers_by_type(markers)
    types_to_drop = []
    for i, cell_type in enumerate(by_type):
        gene_n = len(by_type[cell_type])
        if gene_n <= n:
            types_to_drop.append(cell_type)

    to_drop = []
    for marker in markers:
        c_type = marker[2]
        if c_type in types_to_drop:
            to_drop.append(marker)

    return [marker for marker in markers if marker not in to_drop]


def drop_rows(data, row_names: list):
    """
    Drops rows by label.
    """
    to_keep = backend.get_rows_to_keep(row_names, data.obs['labels'])

    n_dropped = data.shape[0] - len(to_keep)
    data = data[to_keep, :]

    print("Dropped {0} cell(s).".format(n_dropped), "New shape:", data.shape)

    return data


def drop_unused_genes(data, markers, sort_columns=True):
    """
    Drops every column (gene) not found in the marker data set and sorts the columns by name.
    """
    sorted_by_type = backend.sort_markers_by_type(markers)
    used_genes = backend.get_used_genes(sorted_by_type)

    n_dropped = data.shape[1] - len(used_genes)
    data = data[:, used_genes]

    if sort_columns:
        data = data[:, np.argsort(data.var_names)]

    print("Dropped {0} gene(s).".format(n_dropped), "New shape:", data.shape)
    return data


def check_labels(data_sets: list, markers: list, aliases: dict, throw_exception=True) -> None:
    """
    Checks for any cell types (labels) not found in the marker data set.
    """
    unknown = []
    used_types = backend.get_cell_types(markers)

    for data_set in data_sets:
        labels = data_set.obs['labels']
        for unknown_label in backend.check_for_unknown(labels.unique(), used_types, aliases):
            if unknown_label not in unknown:
                unknown.append(unknown_label)

    if len(unknown) != 0:
        if throw_exception:
            raise NameError("Unknown cell type(s)!", unknown)
        else:
            print("====UNKNOWN CELL TYPES====")
            print(unknown)
            print("==========================")


def check_shape(data_sets: list, throw_exception=True) -> None:
    """
    Checks if the there are the same number of genes
    :param data_sets: list of DataFrames to check
    :param throw_exception: throw exception if shapes do not match
    """
    num_genes = data_sets[0].shape[1]
    for data_set in data_sets[1:]:
        if num_genes != data_set.shape[1]:
            if throw_exception:
                raise ValueError("Column lengths do not match!", num_genes, len(data_set.columns))
            else:
                print("COLUMN LENGTHS DO NOT MATCH!", num_genes, len(data_set.columns))

    genes = data_sets[0].var_names.values
    for data_set in data_sets[1:]:
        if np.sum(genes == data_set.var_names.values) - len(genes) != 0:
            if throw_exception:
                raise ValueError("Columns are not in the same order!")
            else:
                print("Columns are not in the same order!")
