import os

import numpy as np
import pandas as pd

import data
import helpers


def split(features, labels, set_distribution=data.SET_DISTRIBUTION):
    datasets = list(zip(helpers.perc_split(features, set_distribution), helpers.perc_split(labels, set_distribution)))
    return datasets


def convert_boxes(img_count, boxes_raw):
    anchors = 1
    new_boxes = np.zeros((img_count, data.STRIDE_W, data.STRIDE_H, anchors * 6))
    for box_raw in boxes_raw.values:
        img_id = box_raw[0]
        if img_count == img_id:
            break
        box = box_raw[1:]
        box[0] = box[0] + box[2] / 2
        box[1] = box[1] + box[3] / 2

        padded = np.c_[1, box.reshape(1, 4), 1]
        new_boxes[int(img_id), int(box[0] / data.STRIDE), int(box[1] / data.STRIDE)] = np.tile(padded, anchors)
    return new_boxes


def preprocess_ego(data_path=data.DATA_PATH):
    images_raw = pd.read_csv(os.path.join(data_path, data.RAW_IMAGES), nrows=data.N_ROWS, header=None, index_col=None,
                             na_filter=False,
                             dtype=np.float32, low_memory=data.LOW_MEMORY)
    images = images_raw.values[:, 3:]

    print('Loaded raw images')

    # boxes_raw.groupby(by=0).size().max() ~ 4
    boxes_raw = pd.read_csv(os.path.join(data_path, data.RAW_LABELS), header=None, index_col=None, na_filter=False,
                            nrows=data.N_ROWS,
                            dtype=np.float32,
                            low_memory=data.LOW_MEMORY)

    boxes = convert_boxes(images.shape[0], boxes_raw)

    return split(*helpers.unison_shuffled_copies(images.astype(np.float32), boxes.astype(np.float32)))


def get_dataset(data_path=data.DATA_PATH):
    return preprocess_ego(data_path)
