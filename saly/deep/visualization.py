import matplotlib.pyplot as plt


def draw_model_history(history):
    """
    Draws a model's training history.
    :param history: a Keras history object
    """

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')

    if 'val_loss' in history.history.keys():
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'b', label='Validation loss', c='orange')
        plt.title('Training and validation loss per epoch')
    else:
        plt.title('Training loss per epoch')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


