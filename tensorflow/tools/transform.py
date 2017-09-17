#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
from PIL import Image


def dense_to_one_hot(labels, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def reshape(data, width, height, channel):
    """format: [num, width, height, channel]"""
    return data.reshape([-1, channel, width, height]).transpose(0, 2, 3, 1)


def resize(data, width, height, channel, width_new, height_new):
    res_data = []
    for _ in data:
        image = (reshape(_, width, height, channel)[0:1, :, :, :]).reshape([width, height, channel])
        img_obj = Image.fromarray(image)
        # img_obj.show()
        img_obj = img_obj.resize(size=(width_new, height_new), resample=Image.LANCZOS)
        # img_obj.show()
        img_array = np.asarray(img_obj, dtype=np.uint8)
        res_data.append(img_array)
    return res_data
