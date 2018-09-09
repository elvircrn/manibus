import os

N_CLASSES = 1
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
SET_DISTRIBUTION = [0.94, 0.03, 0.03]

MODEL_DIR = os.path.relpath('data/log0')

DEFAULT_SCOPE = 'ManibusConv'

RAW_IMAGES = 'images.csv'
RAW_LABELS = 'boxes.csv'
DATA_PATH = 'data/egohands_data'

TRAINING_SCOPE = 'Training_data'
TEST_SCOPE = 'Test_data'

LABELS = [str(i) for i in range(N_CLASSES)]

N_ROWS = 64
LOW_MEMORY = False
