import matplotlib.pyplot as plt
import numpy as np
from seaborn import distplot
from sklearn import metrics
from scipy.special import softmax
from pandas import Series
import math

from .. import backend


def plot_marker_genes(markers, partially_dense=False):
    """
    Draws a bar chart of marker genes per cell type.
    """
    by_type = backend.sort_markers_by_type(markers)

    hist_data = []
    for i, cell_type in enumerate(by_type):
        gene_n = len(by_type[cell_type])
        if partially_dense:
            gene_n = round(np.log2(gene_n))
        hist_data.append(gene_n)

    hist_mean = np.mean(hist_data)
    hist_data = sorted(hist_data, reverse=True)
    ticks = list(range(len(by_type)))

    plt.bar(ticks, hist_data, width=1.0)
    plt.axhline(hist_mean, c='r', label='Mean ({0})'.format(int(np.round(hist_mean))))
    plt.legend()

    title = 'Number of marker genes per cell type'
    if partially_dense:
        title = 'Number of nodes in the partially dense layer per cell type'
        title += '\nSize of partially-dense layer: {0}'.format(int(sum(hist_data)))
    plt.title(title)


def get_predictions(cell_activations, markers):
    """
    Returns the predicted classes.
    """
    cell_types = backend.get_cell_types(markers)

    top_activations = backend.get_top_activated_indices(1, cell_activations)
    predictions = backend.index_to_cell_type(top_activations, cell_types)

    return [prediction[0] for prediction in predictions]


def get_results(labels: list, cell_activations: list, markers: list, aliases: dict) -> float:
    """
    Compares the real labels to the cell type activations.
    :param labels: Data labels
    :param cell_activations: Activations of the Marker Layer nodes
    :param markers: list of used markers
    :param aliases: dict of `label : name_in_marker_db` aliases
    :return
    """

    predictions = get_predictions(cell_activations, markers)
    by_type = backend.sort_markers_by_type(markers)

    correct_types = {}
    correct = 0
    n = len(cell_activations)
    for i, prediction in enumerate(predictions):
        label = labels[i]
        
        if label not in correct_types.keys():
            correct_types[label] = 0

        if prediction == label:
            correct_types[label] += 1
            correct += 1

        elif label in aliases.keys():
            if aliases[label] == prediction:
                correct_types[label] += 1
                correct += 1
    
    print("Correct predictions: {c} out of {n} ({p}%)".format(c=correct, n=n, p=round(100 * (correct / n), 2)))
    
    label_counts = labels.value_counts()
    for c_type in correct_types:
        if c_type in aliases.keys():
            n_markers = len(by_type[aliases[c_type]])
        else:
            n_markers = len(by_type[c_type])
        
        c = round(100 * (correct_types[c_type] / label_counts[c_type]), 2)
        print("\t{}: {}% ({}/{}) | Markers: {}".format(
            c_type, c, correct_types[c_type], label_counts[c_type], n_markers))

    return (correct / n)


def plot_model_history(history, baseline_val_acc=None, supervised=False, labelled_training=False):
    """
    Draws a model's training history.
    :param baseline_val_acc: accuracy of the baseline model on the validation set
    :param history: a Keras history object
    :param supervised: was the model supervised
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(7, 7), dpi=80)
    output_loss = history.history['output_loss']

    accuracy = history.history['cell_activations_marker_prediction_metric']
    accuracy = [i * 100 for i in accuracy]

    epochs = range(1, len(output_loss) + 1)

    if supervised:
        output_loss = [i * 100 for i in output_loss]
        marker_loss = history.history['cell_activations_loss']
        ax1.plot(epochs, marker_loss, 'b-,', label='Cell type prediction loss')

    ax1.plot(epochs, output_loss, 'b--', label='Reconstruction loss')
    
    if labelled_training:
        ax2.plot(epochs, accuracy, 'g--', label='Training accuracy')

    if 'val_loss' in history.history.keys():
        val_output_loss = history.history['val_output_loss']

        val_accuracy = history.history['val_cell_activations_marker_prediction_metric']
        val_accuracy = [i * 100 for i in val_accuracy]

        if supervised:
            val_output_loss = [i * 100 for i in val_output_loss]
            val_marker_loss = history.history['val_cell_activations_loss']
            ax1.plot(epochs, val_marker_loss, 'r-', label='Validation cell type prediction loss')

        ax1.plot(epochs, val_output_loss, 'r--', label='Validation reconstruction loss')
        ax1.set_title('Training and validation loss')

        ax2.plot(epochs, val_accuracy, 'g-', label='Validation accuracy')

        if labelled_training:
            ax2.set_title('Training, validation and baseline accuracy')
        else:
            ax2.set_title('Validation and baseline accuracy')

        if baseline_val_acc is not None:
            baseline_val_acc = baseline_val_acc * 100 if baseline_val_acc <= 1. else baseline_val_acc
            ax2.axhline(baseline_val_acc, c='r', label=f'Baseline accuracy ({round(baseline_val_acc, 2)}%)')
    else:
        ax1.set_title('Training loss per epoch')
        ax2.set_title('Training accuracy per epoch')

    fig.text(0.04, 0.5, 'Accuracy (%)                                                    Loss', va='center',
             rotation='vertical')
    plt.xlabel('Epochs')
    ax1.legend()
    ax2.legend()

    plt.show()


def plot_activation_distribution(cell_activations, markers, title=''):
    """
    Plots the average cell type activation.
    :param cell_activations: Marker layer node activations
    :param markers: list of used markers
    :param title: graph title
    """
    n_cell_types = len(backend.get_cell_types(markers))
    sorted_indices = backend.get_top_activated_indices(n_cell_types, cell_activations)

    sorted_activations = []
    for i, cell in enumerate(cell_activations):
        sorted_activations.append([cell[j] for j in sorted_indices[i]])

    average = [np.mean(column) for column in np.transpose(sorted_activations)]

    distplot(average).set_title(title)


def get_label_colours(labels):
    """
    Generates a dictionary of colours for each label
    :param labels: list of labels
    :return: dictionary of label : colour
    """
    colours = {}
    for label in labels.unique():
        colour = (backend.get_random_colour())
        colours.update({label: colour})

    return colours


def plot_label_colours(colours):
    """
    Plots a list of colours
    :param colours:
    """
    if type(colours) == dict:
        colours = colours.values()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = list(range(0, len(colours) * 2, 2))
    y = [0] * len(x)
    c = list(colours)

    ax.scatter(x, y, s=400, c=c)
    ax.set_title('Cell type colours')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def get_top_activations(n, cell_activations):
    """
    Returns cell activations with only the top n activations
    :param n: top number of cell types to return
    :param cell_activations: list of cell activations
    """
    top_indices = backend.get_top_activated_indices(n, cell_activations)
    top_activations = [np.zeros(len(x)) for x in cell_activations]
    for i, cell_indices in enumerate(top_indices):
        for type_index in cell_indices:
            top_activations[i][type_index] = cell_activations[i][type_index]

    return top_activations


def draw_embedding(x, y, model, colours=None, alpha=1.0, graph_title=''):
    """
    Draws a scatter plot of the provided embedding model
    :param x: Data x
    :param y: Data labels
    :param model: embedder e.g. tSNE or PCA
    :param colours: label colours
    :param alpha: Node alpha
    :param graph_title: Graph title
    """
    if colours is None:
        colours = get_label_colours(y)

    model_out = model.fit_transform(x)

    plt.figure(figsize=(8, 8), dpi=80)
    for i, point in enumerate(model_out):
        plt.scatter(point[0], point[1], color=colours[y.iloc[i]], label=y.iloc[i],
                    alpha=alpha, edgecolors='black', linewidths=1.0)

    # Create a legend
    plt_handles, plt_labels = plt.gca().get_legend_handles_labels()
    handles, labels = backend.get_graph_labels(plt_handles, plt_labels)

    for handle in handles:
        handle.set_alpha(1.0)

    plt.legend(handles, labels, bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)

    plt.title(graph_title)
    plt.axis('off')
    plt.show()


def compare_embeddings(data_1, data_2, model, colours=None, alpha=1.0, graph_titles=None):
    """
    Draws a scatter plot of the provided embedding model
    :param data_1: First data set -- list of the data's x and y
    :param data_2: Second data set -- list of data's x and y
    :param model: embedder e.g. tSNE or PCA
    :param colours: label colours
    :param alpha: Node alpha
    :param graph_titles: list of graph titles
    """
    x1, y1 = data_1[0], data_1[1]
    x2, y2 = data_2[0], data_2[1]

    if colours is None:
        colours = get_label_colours(y1.append(y2))

    model_out_1 = model.fit_transform(x1)
    model_out_2 = model.fit_transform(x2)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 16), dpi=120)
    ax1.set(aspect='equal')
    ax2.set(aspect='equal')

    for i, point in enumerate(model_out_1):
        ax1.scatter(point[0], point[1], color=colours[y1.iloc[i]], label=y1.iloc[i],
                    alpha=alpha, edgecolors='black', linewidths=1.0)
    for i, point in enumerate(model_out_2):
        ax2.scatter(point[0], point[1], color=colours[y2.iloc[i]], label=y2.iloc[i],
                    alpha=alpha, edgecolors='black', linewidths=1.0)

    # Create a legend
    plt_handles, plt_labels = plt.gca().get_legend_handles_labels()
    handles, labels = backend.get_graph_labels(plt_handles, plt_labels)

    for handle in handles:
        handle.set_alpha(1.0)

    fig.legend(handles, labels, ncol=len(labels), loc=8, bbox_to_anchor=(0.425, 0.1))

    ax1.set_title(graph_titles[0])
    ax1.set_axis_off()

    ax2.set_title(graph_titles[1])
    ax2.set_axis_off()

    plt.subplots_adjust(bottom=0., hspace=0., wspace=0.25)
    plt.show()


def draw_comparison(old, new, model, colours=None, graph_title=''):
    """
    Draws a combined graph using both data sets
    :param old: a list of old data (data, labels)
    :param new: a list of new data (data, labels)
    :param model: embedder e.g. tSNE or PCA
    :param colours: label colours
    :param graph_title: Graph title
    """
    if colours is None:
        colours = get_label_colours(old[1])

    x = np.concatenate((old[0], new[0]), axis=0)
    model_out = model.fit_transform(x)

    plt.figure(figsize=(8, 8), dpi=80)
    for i, point in enumerate(model_out):
        if i < len(old[0]):
            plt.scatter(point[0], point[1], color=colours[old[1].iloc[i]], label=old[1].iloc[i], alpha=0.1)
        else:
            index = i - len(old[0])
            plt.scatter(point[0], point[1], color=colours[new[1].iloc[index]], label=new[1].iloc[index],
                        alpha=1.0, edgecolors='black', linewidths=1)

    # Create a legend
    plt_handles, plt_labels = plt.gca().get_legend_handles_labels()
    handles, labels = backend.get_graph_labels(plt_handles, plt_labels)

    for handle in handles:
        handle.set_edgecolors('black')
        handle.set_alpha(1.0)

    plt.legend(handles, labels, bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)

    plt.title(graph_title)
    plt.axis('off')
    plt.show()


def draw_confusion_matrix(labels, cell_activations, markers, aliases, title=None, cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix'

    cell_types = backend.get_cell_types(markers)
    top_activations = backend.get_top_activated_indices(1, cell_activations)

    predictions = backend.index_to_cell_type(top_activations, cell_types)
    predictions = Series([p[0] for p in predictions])

    y_true = labels
    y_true = y_true.apply(lambda x: aliases[x] if x in aliases.keys() else x)
    unique_classes = sorted(y_true.append(predictions).unique())

    cm = metrics.confusion_matrix(y_true, predictions)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=unique_classes, yticklabels=unique_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


def draw_roc(labels, cell_activations, markers, aliases):
    """
    Returns the average ROC AUC score and, if defined, draws an ROC graph for each class.
    """
    probs = softmax(cell_activations, axis=1)
    evaluation = backend.get_class_evaluation(labels, cell_activations, markers, aliases)

    by_type = backend.sort_markers_by_type(markers)
    types = list(by_type.keys())
    y_true = backend.one_hot_encode(labels, markers, aliases)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(types)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    n = 0
    used = []
    to_draw = []

    for i, c_type in enumerate(types):
        score = roc_auc[i]
        if not np.isnan(score):
            n += 1
            used.append(score)
            to_draw.append([fpr[i], tpr[i], score, c_type])

    # Drawing the plots
    n_columns = 3
    n_rows = int(math.ceil(len(to_draw) / n_columns))

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_columns)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.subplots_adjust(top=0.6, bottom=0.01, hspace=0.2, wspace=0.2)

    k = 0
    for i, col in enumerate(ax):
        for j, row in enumerate(col):
            if k > len(to_draw):
                row.remove()
                continue

            plot = to_draw[k]
            row.plot(plot[0], plot[1], label='ROC curve (area = %0.2f)' % plot[2])
            row.plot([0, 1], [0, 1], 'k--')

            row.set_aspect(1.)
            row.set_xlim([0.0, 1.0])
            row.set_ylim([0.0, 1.05])

            f1 = evaluation[plot[3]]['f1']
            row.plot([], [], ' ', label=f'F1 score: {round(f1, 3)}')

            row.set_xlabel('False Positive Rate')
            row.set_ylabel('True Positive Rate')
            row.title.set_text(f'ROC curve of {plot[3]}')
            row.legend(loc="lower right")

            k += 1

    plt.show()
