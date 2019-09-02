from .. import backend
from Orange.data import Table
from anndata import read_h5ad
from pandas import DataFrame, Series
from numpy import sum


def load_h5ad(path: str) -> (DataFrame, Series):
    """
    Loads a data set from the h5ad format as a DataFrame.
    :param path: The path to the .h5ad file
    :return: A DataFrame (data) and a Series (labels) object
    """
    dataset = read_h5ad(path)
    labels = dataset.obs["labels"]

    return dataset, labels


def load_markers(marker_path: str, species: str) -> list:
    """
    Loads marker genes of the specified species.
    :param marker_path: The path to the tab separated file
    :param species: "Mouse", "Human" etc.
    :return: List of markers
    """
    markers_db = Table(marker_path)
    markers_db = backend.get_markers_by_species(species, markers_db)

    return markers_db


def get_mutual_markers(gene_sets: list, markers_db: list) -> list:
    """
    :param gene_sets: lists of used genes
    :param markers_db: a list of markers
    :return: an intersection of markers used in the provided data set(s).
    """
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


def drop_rows(data: DataFrame, labels: Series, row_names: list) -> (DataFrame, Series):
    """
    :param data: The DataFrame to modify
    :param labels: Series of labels
    :param row_names: a list of row labels to drop
    :return: the modified DataFrame and Labels
    """
    to_drop = backend.get_rows_to_drop(row_names, labels)

    data = data.drop(data.index[to_drop], axis=0)
    labels = labels.drop(labels.index[to_drop], axis=0)

    print("Dropped {0} cell(s).".format(len(to_drop)), "New shape:", data.shape)

    return data, labels


def drop_unused_genes(data: DataFrame, markers: list, sort_columns=True) -> DataFrame:
    """
    Drops every column (gene) not found in the marker data set and sorts the columns by name.
    :param data: The DataFrame to modify
    :param markers: list of used gene markers
    :param sort_columns: if True sorts columns by name
    :return: The modified DataFrame
    """
    sorted_by_type = backend.sort_markers_by_type(markers)
    used_genes = backend.get_used_genes(sorted_by_type)

    unused_genes = [gene for gene in data.columns if gene not in used_genes]
    data = data.drop(unused_genes, axis=1)

    if sort_columns:
        data = data.reindex(sorted(data.columns), axis=1)

    print("Dropped {0} gene(s).".format(len(unused_genes)), "New shape:", data.shape)
    return data


def check_labels(label_sets: list, markers: list, aliases: dict, throw_exception=True) -> None:
    """
    Checks for any cell types (labels) not found in the marker data set.
    :param label_sets: lists of used labels
    :param markers: list of markers
    :param aliases: dictionary of `label : name_in_marker_db`
    :param throw_exception: throw exception if unknown labels were found
    """
    unknown = []
    used_types = backend.get_cell_types(markers)

    for labels in label_sets:
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
    Checks if the there are the same number of
    :param data_sets: list of DataFrames to check
    :param throw_exception: throw exception if shapes do not match
    """
    num_genes = len(data_sets[0].columns)
    for data_set in data_sets[1:]:
        if num_genes != len(data_set.columns):
            if throw_exception:
                raise ValueError("Column lengths do not match!", num_genes, len(data_set.columns))
            else:
                print("COLUMN LENGTHS DO NOT MATCH!", num_genes, len(data_set.columns))

    genes = data_sets[0].columns.values
    for data_set in data_sets[1:]:
        if sum(genes == data_set.columns.values) - len(genes) != 0:
            if throw_exception:
                raise ValueError("Columns are not in the same order!")
            else:
                print("COLUMNS ARE NOT IN THE SAME ORDER!")
