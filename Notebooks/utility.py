from collections import OrderedDict

################################
# MARKER GENE HELPER FUNCTIONS #
################################

"""
Returns an array of marker genes of the
specified species.
"""
def get_markers_by_species(species, markers):
    markers_by_species = []
    for meta in markers.metas:
        if meta[0] == species:
            markers_by_species.append(meta)
    return markers_by_species

"""
Returns the marker genes found in the given dataset.
"""
def get_used_markers(data, markers):
    used_markers = []
    for i, gene_name in enumerate(data.columns):
        for marker in markers:
            if gene_name == marker[1]:
                used_markers.append((gene_name, marker))
    return used_markers

"""
Returns an OrderedDictionary of every cell type and its
corresponding genes found in the given array of markers.
"""
def sort_markers_by_type(markers):
    by_type = OrderedDict()
    for marker in markers:
        c_type = marker[1][2]
        gene_name = marker[0]
        
        # check if the key already exists
        if c_type in by_type: 
            # check if the gene is already in the array
            if gene_name not in by_type[c_type]: 
                by_type[c_type].append(gene_name)
                
        else: # create a new entry
            by_type[c_type] = [gene_name]
    return by_type

"""
Returns a (unique) array of genes used
in the specified array.
"""
def get_used_genes(markers_by_type):
    used_genes = []
    for c_type in markers_by_type:
        genes = markers_by_type[c_type]
        for gene in genes:
            if gene not in used_genes:
                used_genes.append(gene)
    return used_genes

"""
Returns an array of cell types given an
array of markers sorted by cell type.
"""
def get_cell_types(markers_by_type):
    used_types = []
    for c_type in markers_by_type:
        used_types.append(c_type)
    return used_types

"""
Returns the markers found in both provided datasets.
"""
def get_mutual_markers(dataset1, dataset2):
    mutual = []
    for marker in dataset1:
        marker_found = False
        for new_marker in dataset2:
            if marker[0] == new_marker[0] and marker[1][2] == new_marker[1][2]:
                mutual.append(marker)
    return mutual

"""
Prints out any cell type not found in either of the 
provided arrays.
"""
def check_for_unknown(labels, used_types, aliases):
    for c_type in labels.unique():
        if c_type not in used_types and c_type not in list(aliases.keys()):
            print(c_type)
            
            
###################################
# GENE SELECTION HELPER FUNCTIONS #
###################################

"""
Returns a list of row indicies to drop from a list
of row labels.
"""
def get_rows_to_drop(labels_to_drop, labels):
    to_drop = []
    for i, c_type in enumerate(labels):
        if c_type in labels_to_drop:
            to_drop.append(i)
    return to_drop

"""
Returns a list of column indicies to drop from a list
of columns.
"""
def get_columns_to_drop():
    to_drop = []
    return to_drop