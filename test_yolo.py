import itertools

import cv2
import data
import matplotlib.pyplot as plt
import numpy as np
import data
from PIL import Image
import tensorflow as tf

import network as net


def draw(img_hand):
    Image.fromarray(img_hand.reshape(28, 28) * 255).show()


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


if __name__ == '__main__':
    tf.enable_eager_execution()
    cap = cv2.VideoCapture(0)

    net.initialize_flags(model_dir='data')
    estimator = net.get_estimator()
    cv2.startWindowThread()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (256, 144))

        prediction = next(net.predict(estimator, gray.T / 255))

        for i in range(data.STRIDE_W):
            for j in range(data.STRIDE_H):
                p = prediction[i][j]
                if p[4] > 0.6:
                    x = i * data.STRIDE + p[0] * data.STRIDE
                    y = j * data.STRIDE + p[1] * data.STRIDE
                    w = p[2] * data.IMAGE_WIDTH
                    h = p[3] * data.IMAGE_HEIGHT

                    cv2.rectangle(gray, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), 2)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
