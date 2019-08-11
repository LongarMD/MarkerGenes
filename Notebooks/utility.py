
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