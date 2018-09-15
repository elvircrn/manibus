import os

N_CLASSES = 1
IMAGE_WIDTH, IMAGE_HEIGHT = 256, 144
STRIDE = 16
STRIDE_W, STRIDE_H = int(IMAGE_WIDTH / STRIDE) + 1, int(IMAGE_HEIGHT / STRIDE) + 1

SET_DISTRIBUTION = [0.94, 0.03, 0.03]

MODEL_DIR = os.path.relpath('data/log0')

DEFAULT_SCOPE = 'ManibusConv'

RAW_IMAGES = 'images_020.csv'
RAW_LABELS = 'boxes_020.csv'
DATA_PATH = 'data/egohands_data'

TRAINING_SCOPE = 'Training_data'
TEST_SCOPE = 'Test_data'

LABELS = [str(i) for i in range(N_CLASSES)]

N_ROWS = 64
LOW_MEMORY = False
