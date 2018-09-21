import os

import numpy as np
import pandas as pd

import data
import helpers


def split(faces, labels, set_distribution=data.SET_DISTRIBUTION):
    datasets = list(zip(helpers.perc_split(faces, set_distribution), helpers.perc_split(labels, set_distribution)))
    return datasets


def preprocess(data_path=data.DATA_PATH):
    train_df = pd.read_csv(data_path + os.sep + data.HANDS_TRAIN_NAME)
    test_df = pd.read_csv(data_path + os.sep + data.HANDS_TEST_NAME)

    dataset = train_df.append(test_df).as_matrix().astype(np.float32)

    labels = dataset.T[0]
    features = (dataset.T[1:]).T

    features = features / features.max()

    features, labels = helpers.unison_shuffled_copies(features, labels)
    one_hot_labels = np.zeros((len(labels), data.HANDS_N_CLASSES), dtype=np.float32)
    print(one_hot_labels[0][0])
    print(one_hot_labels.shape)
    one_hot_labels[np.arange(len(labels)), labels.astype(np.int32)] = 1
    datasets = split(features, one_hot_labels)

    return datasets


def get_dataset():
    return preprocess()
