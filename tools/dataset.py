#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import numpy as np
import os


def load_cifar10(file_path, one_hot=False, num_classes=None, test_rate=0.3):
    files = os.listdir(file_path)
    features = []
    labels = []
    for file in files:
        file_p = os.path.join(file_path, file)
        fea, lab = load(file_p)
        features.append(fea)
        labels.append(lab)
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    if one_hot:
        labels = dense_to_one_hot(np.asarray(labels), num_classes)

    len = features.shape[0]
    features_test = features[:int(test_rate*len), :]
    features_train = features[int(test_rate*len):len, :]
    labels_test = labels[:int(test_rate*len), :]
    labels_train = labels[int(test_rate*len):len, :]

    return features_train, labels_train, features_test, labels_test


def load(file):
    major = sys.version_info.major
    dict = {}
    if major == 2:
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
    elif major == 3:
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    features, labels = dict[b'data'], dict[b'labels']
    return features, labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
