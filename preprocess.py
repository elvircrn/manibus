import pandas as pd
import numpy as np
import helpers
import data

def preprocess():
    train_df = pd.read_csv(data.TRAIN_DATASET_PATH)
    test_df = pd.read_csv(data.TEST_DATASET_PATH)

    dataset = train_df.append(test_df)

    print()

    dataset = dataset.shuffle()
    dataset = helpers.perc_split(dataset, data.percentages) 
    labels = (np.matrix(dataset.as_matrix().astype(np.float32)).T)[0]
    features = ((dataset.T)[1:]).T

    helpers.unison_shuffled_copies(dataset, data.SET_DISTRIBUTION)


def get_dataset():
    return preprocess()


