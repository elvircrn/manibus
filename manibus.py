import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import network


def experiment():
    images = pd.read_csv('data/egohands_data/images.csv', nrows=50, header=None, index_col=None, na_filter=False,
                         dtype=np.float64, low_memory=False).values[:, 3:]
    boxes = pd.read_csv('data/egohands_data/boxes.csv', header=None, index_col=None, na_filter=False,
                        dtype=np.float64, low_memory=False).values[:, 1:]

    img = Image.fromarray(images[0].reshape(640, 360).T * 255).convert('LA')
    draw = ImageDraw.Draw(img)
    print(boxes[0])
    print(boxes.shape)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    draw.rectangle(boxes[0])
    img.save('image.png')


# ImageDraw.Draw(resized)

# plt.imshow(images[0].reshape(640, 360))

network.run_network()
