import matplotlib.pyplot as plt
from numpy import concatenate, mean, transpose, zeros, arange, newaxis, isnan
from seaborn import distplot
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.special import softmax
from pandas import Series

from .. import backend


def get_results(labels: list, cell_activations: list, markers: list, aliases: dict) -> None:
    """
    Compares the real labels to the cell type activations.
    :param labels: Data labels
    :param cell_activations: Activations of the Marker Layer nodes
    :param markers: list of used markers
    :param aliases: dict of `label : name_in_marker_db` aliases
    """
    cell_types = backend.get_cell_types(markers)

    top_activations = backend.get_top_activated_indices(1, cell_activations)
    predictions = backend.index_to_cell_type(top_activations, cell_types)

    correct = 0
    n = len(cell_activations)
    for i, prediction in enumerate(predictions):
        label = labels[i]
        prediction = prediction[0]  # because predictions is a 2D array

        if prediction == label:
            correct += 1

        elif label in aliases.keys():
            if aliases[label] == prediction:
                correct += 1

    print("Correct predictions: {c} out of {n} ({p}%)".format(c=correct, n=n, p=round(100 * (correct / n), 2)))


def plot_model_history(history):
    """
    Draws a model's training history.
    :param history: a Keras history object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(7, 7), dpi=80)
    output_loss = history.history['output_loss']
    output_loss = [i * 100 for i in output_loss]

    accuracy = history.history['cell_activations_marker_prediction_metric']
    accuracy = [i * 100 for i in accuracy]

    marker_loss = history.history['cell_activations_loss']
    epochs = range(1, len(output_loss) + 1)

    ax1.plot(epochs, output_loss, 'b--', label='Reconstruction loss')
    ax1.plot(epochs, marker_loss, 'b-,', label='Cell type prediction loss')
    ax2.plot(epochs, accuracy, 'g--', label='Training accuracy')

    if 'val_loss' in history.history.keys():
        val_output_loss = history.history['val_output_loss']
        val_output_loss = [i * 100 for i in val_output_loss]

        val_accuracy = history.history['val_cell_activations_marker_prediction_metric']
        val_accuracy = [i * 100 for i in val_accuracy]

        val_marker_loss = history.history['val_cell_activations_loss']

        ax1.plot(epochs, val_output_loss, 'r--', label='Validation reconstruction loss')
        ax1.plot(epochs, val_marker_loss, 'r-', label='Validation cell type prediction loss')
        ax1.set_title('Training and validation loss')

        ax2.plot(epochs, val_accuracy, 'g-', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
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

    average = [mean(column) for column in transpose(sorted_activations)]

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
    top_activations = [zeros(len(x)) for x in cell_activations]
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

    x = concatenate((old[0], new[0]), axis=0)
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


def draw_confusion_matrix(y_true, cell_type_activations, markers, aliases,
                          normalize=False, title=None, cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    cell_types = backend.get_cell_types(markers)
    top_activations = backend.get_top_activated_indices(1, cell_type_activations)

    predictions = backend.index_to_cell_type(top_activations, cell_types)
    predictions = Series([p[0] for p in predictions])

    y_true = y_true.apply(lambda x: aliases[x] if x in aliases.keys() else x)
    unique_classes = y_true.append(predictions).unique()

    cm = confusion_matrix(y_true, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=arange(cm.shape[1]),
           yticks=arange(cm.shape[0]),
           xticklabels=unique_classes, yticklabels=unique_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def get_average_auc(labels, cell_activations, markers, aliases, draw=False):
    """
    Returns the average ROC AUC score and, if defined, draws an ROC graph for each class.
    """
    probs = softmax(cell_activations)

    by_type = backend.sort_markers_by_type(markers)
    types = list(by_type.keys())
    y_true = backend.one_hot_encode(labels, markers, aliases)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(types)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    n = 0
    used = []
    for i in range(n_classes):
        score = roc_auc[i]

        if isnan(score) == False:
            n += 1
            used.append(score)
            if draw:
                plt.figure()
                plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % score)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.show()
    print("Average ROC AUC:", round(sum(used) / n, 3))
