import pandas as pd
import numpy as np
import helpers

TRAIN_DATASET_PATH = 'data/sign_mnist_train.csv'
TEST_DATASET_PATH = 'data/sign_mnist_test.csv'

train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)

dataset = train_df.append(test_df)

print()

dataset = dataset.shuffle()
dataset = helpers.perc_split(dataset, data.percentages)


labels = (np.matrix(dataset.as_matrix().astype(np.float32)).T)[0]
features = ((dataset.T)[1:]).T


