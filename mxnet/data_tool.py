# -*- coding:utf-8 -*-


import os
import numpy as np
import sys
# from PIL import Image


def load_data(traindata_file_path, testdata_file_path, one_hot=True, classes=2):
    traindata, trainlabel = read_file(traindata_file_path)
    testdata, testlabel = read_file(testdata_file_path)

    if one_hot:
        trainlabel = onehot(trainlabel, classes)
        testlabel = onehot(testlabel, classes)

    return traindata, trainlabel, testdata, testlabel


def read_file(file_path):
    file_list = os.listdir(file_path)

    features = []
    labels = []

    for file in file_list:
        with open(os.path.join(file_path, file), 'rb') as fb:
            major = sys.version_info.major
            data = {}
            if major == 2:
                import cPickle
                data = cPickle.load(fb)
            elif major == 3:
                import pickle
                data = pickle.load(fb, encoding='bytes')
            features.append(data['data'])
            labels.append(data['label'])

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels


def onehot(ndarr, classes):
    num_labels = ndarr.shape[0]
    index_offset = np.arange(num_labels) * classes
    labels_one_hot = np.zeros((num_labels, classes))
    labels_one_hot.flat[index_offset + ndarr.ravel()] = 1
    return labels_one_hot
