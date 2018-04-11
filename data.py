import os

N_CLASSES = 26
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
SET_DISTRIBUTION = [0.94, 0.03, 0.03]
TRAIN_DATASET_PATH = 'data/sign_mnist_train.csv'
TEST_DATASET_PATH = 'data/sign_mnist_test.csv'


# TODO: Start using relative path
MODEL_DIR = os.path.abspath("data/log")

DEFAULT_SCOPE = 'ManibusConv'

TRAINING_SCOPE = 'Training_data'
TEST_SCOPE = 'Test_data'

LABELS = [str(i) for i in range(N_CLASSES)]

