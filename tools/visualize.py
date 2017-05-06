#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def vis_square(data, padsize=1, padval=0):
    """take an array of shape (n, height, width) or (n, height, width, channels)
        and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='nearest', cmap=cm.binary)
    plt.show()


def get_time():
    import time
    time_format = '%Y-%m-%d--%H-%M-%S'
    return time.strftime(time_format, time.localtime())
