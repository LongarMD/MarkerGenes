from .data import sort_markers_by_type

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import activations, initializers, regularizers, constraints
from keras.activations import softmax
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.engine.base_layer import InputSpec

def get_partially_dense_size(by_type):
    """
    Returns the number of nodes in the partially dense layer.
    """
    nodes = []
    for i, cell_type in enumerate(by_type):
        gene_n = len(by_type[cell_type])
        gene_n = round(np.log2(gene_n))
        nodes.append(gene_n)

    return int(np.sum(nodes))


def get_partially_dense_mask(by_cell_type, genes):
    """
    Creates a binary mask for the partially dense layer.
    """
    node_dim = get_partially_dense_size(by_cell_type)
    mask = np.zeros(shape=(len(genes), node_dim))

    i = 0
    for cell_type in by_cell_type:
        marker_genes = by_cell_type[cell_type]
        N = len(marker_genes)
        n = int(round(np.log2(N)))

        for node in range(n):
            for gene in by_cell_type[cell_type]:
                gene_index = genes.get_loc(gene)
                mask[gene_index][i] = 1.0
            i += 1

    return mask


def get_marker_mask(by_cell_type):
    """
    Creates a binary mask for the marker layer.
    :param by_cell_type: Markers sorted by cell type
    """
    node_dim = get_partially_dense_size(by_cell_type)
    mask = np.zeros(shape=(node_dim, len(by_cell_type)))

    i = 0
    for c, cell_type in enumerate(by_cell_type):
        marker_genes = by_cell_type[cell_type]
        N = len(marker_genes)
        n = int(round(np.log2(N)))

        for node in range(n):
            mask[i][c] = 1.0
            i += 1
    return mask


def get_weight_mask(shape, by_cell_type, genes):
    """
    Deprecated.
    Creates a binary mask for the marker layer
    :param shape: shape of the matrix (n of cell types, n of genes)
    :param by_cell_type: Markers sorted by cell type
    :param genes: list of used genes (in the same order as in the data)
    """
    mask = np.zeros(shape=shape)
    for i, cell_type in enumerate(by_cell_type):
        for gene in by_cell_type[cell_type]:
            gene_index = genes.get_loc(gene)
            mask[i][gene_index] = 1.0
    return mask


def one_hot_encode(labels, markers, aliases):
    """
    One hot encodes a list of marker cell types
    :param labels: Used labels
    :param markers: Used markers
    :param aliases: Cell type aliases (saly.check_labels)
    """
    by_type = sort_markers_by_type(markers)
    types = list(by_type.keys())

    one_hot = np.zeros(shape=(len(labels), len(by_type)))
    for i, label in enumerate(labels):
        if label == -1:
            one_hot[i] = np.repeat(-1, len(by_type))
        else:
            if label in types:
                label_index = types.index(label)
            elif label in aliases.keys():
                label_index = types.index(aliases[label])
            else:
                raise NameError("Unknown cell type!", label)
    
            one_hot[i][label_index] = 1.0

    return one_hot


def marker_loss(y_true, y_pred):
    """
    Get the marker cell type activations classification loss
    """
    probs = softmax(y_pred)
    return categorical_crossentropy(y_true, probs)


def null_loss(y_true, y_pred):
    """
    An empty loss function.
    """
    return 0 * y_true


def celltype_accuracy(y_true, y_pred):    
    return categorical_accuracy(y_true, y_pred)


class Partial(Layer):

    def __init__(self, units, weight_mask,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Partial, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        weight_mask = tf.convert_to_tensor(weight_mask, dtype=tf.float32)
        self.weight_mask = weight_mask

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        conns = tf.multiply(self.kernel, self.weight_mask)
        output = K.dot(inputs, conns)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Partial, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
