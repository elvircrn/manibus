import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

import network_hands as net
import preprocess_hands
import data

def get_image(img_hand):
    return Image.fromarray(img_hand.reshape(28, 28) * 255)


def draw(img_hand):
    get_image(img_hand).show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_samples(predictions, labels, hands):
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.text(-4, 36,
                'Labela: ' + str(chr(ord("A") + labels[i][0])) + ' Predikcija: ' + str(chr(ord("A") + predictions[i])))
        plt.imshow(get_image(hands[i]))


if __name__ == '__main__':
    net.initialize_flags()
    estimator = net.get_estimator()

    dataset = preprocess_hands.preprocess()[1]
    test_set = dataset[0]
    test_labels = dataset[1]
    sample_size = 200
    rand_ids = [random.randrange(len(test_set)) for id in range(sample_size)]
    hands = test_set[rand_ids]
    labels = [np.where(label > 0)[0] for label in test_labels[rand_ids]]

    lazy_predictions = net.predict(estimator, hands)
    predictions = [next(lazy_predictions) for i in range(sample_size)]

    # plot_samples(predictions, labels, hands)

    matrix = confusion_matrix(labels, predictions)
    plot_confusion_matrix(matrix, [str(id) for id in range(data.N_CLASSES)])
    plt.show()
