import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import data
import architectures as arch
import network
import tensorflow as tf

def experiment():
    images = pd.read_csv('data/egohands_data/images.csv', nrows=50, header=None, index_col=None, na_filter=False,
                         dtype=np.float32, low_memory=False).values[:, 3:]
    boxes = pd.read_csv('data/egohands_data/boxes.csv', header=None, index_col=None, na_filter=False,
                        dtype=np.float32, low_memory=False).values[:, 1:]

    img = Image.fromarray(images[0].reshape(640, 360).T * 255).convert('LA')
    draw = ImageDraw.Draw(img)
    print(boxes[0])
    print(boxes.shape)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    draw.rectangle(boxes[0])
    img.save('image.png')


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else data.DATA_PATH
    print('Data path: ', data_path)

    network.run_network(data_path)
    # print(
    #     arch.yolo_arch_fast_020([], False, 0.0)
    # )

