#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
    dataset url: https://www.cs.toronto.edu/~kriz/cifar.html
    source format: channel, width, height
"""


import sys
import os
import numpy as np
from tools import transform
import math
import six


_CIFAR10_DATAKEY = b'data'
_CIFAR10_LABELKEY = b'labels'
_CIFAR100_DATAKEY = b'data'
_CIFAR100_FINELABELKEY = b'fine_labels'
_CIFAR100_COARSELABELKEY = b'coarse_labels'


class CifarDataset(object):

    def __init__(self,
                 path='cifar',
                 test_rate=0.3,
                 one_hot=True,
                 reshape=False):
        self._path = path
        self._test_rate = test_rate
        self._one_hot = one_hot
        self._reshape = reshape
        self._cifar10_classes = 10
        self._cifar100_classes = 100
        self._cifar100_super_classes = 20

    def read_file(self, file, datakey, labelkey):
        major = sys.version_info.major
        dict_t = {}
        if major == 2:
            import cPickle
            with open(file, 'rb') as fo:
                dict_t = cPickle.load(fo)
        elif major == 3:
            import pickle
            with open(file, 'rb') as fo:
                dict_t = pickle.load(fo, encoding='bytes')
        if len(labelkey) == 2:
            features, labels, coarse_labels = dict_t[datakey], dict_t[labelkey[0]], dict_t[labelkey[1]]
            return features, labels, coarse_labels
        elif len(labelkey) == 1:
            features, labels = dict_t[datakey], dict_t[labelkey[0]]
            return features, labels
        else:
            raise ValueError('Invalid labelkeys')

    def save_file(self, features, labels, coarse_labels, dir_name, file_name, cifar10=True):
        import pickle
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        with open(dir_name+os.sep+file_name, 'wb') as fo:
            dict_t = {}
            if cifar10:
                dict_t[_CIFAR10_DATAKEY] = features
                dict_t[_CIFAR10_LABELKEY] = labels
            else:
                dict_t[_CIFAR100_DATAKEY] = features
                dict_t[_CIFAR100_FINELABELKEY] = labels
                dict_t[_CIFAR100_COARSELABELKEY] = coarse_labels
            pickle.dump(dict_t, fo, 1)
            fo.close()

    def load_cifar10(self, width, height, channel, path=None):
        """before reshape: num * channel * width * height ;after reshape: [num, width, height, channel]"""
        if not path:
            path = self._path
        files = os.listdir(path)
        if len(files) == 0:
            raise FileNotFoundError('no files found at dir %s' % path)
        features = []
        labels = []
        for file in files:
            file_p = os.path.join(path, file)
            fea, lab = self.read_file(file_p,
                                      datakey=_CIFAR10_DATAKEY,
                                      labelkey=[_CIFAR10_LABELKEY])
            features.append(fea)
            labels.append(lab)
        features = np.concatenate(features)
        labels = np.concatenate(labels)

        if self._one_hot:
            labels = transform.dense_to_one_hot(labels, self._cifar10_classes)

        size = features.shape[0]

        features_test = features[:int(self._test_rate * size), :]
        features_train = features[int(self._test_rate * size):size, :]
        if self._one_hot:
            labels_test = labels[:int(self._test_rate * size), :]
            labels_train = labels[int(self._test_rate * size):size, :]
        else:
            labels_test = labels[:int(self._test_rate * size)]
            labels_train = labels[int(self._test_rate * size):size]

        if self._reshape:
            features_train = transform.reshape(features_train, width, height, channel)
            features_test = transform.reshape(features_test, width, height, channel)

        return features_train, labels_train, features_test, labels_test

    def load_cifar100(self, width, height, channel, path=None):
        """before reshape: num * channel * width * height ;after reshape: [num, width, height, channel]"""
        if not path:
            path = self._path
        files = os.listdir(path)
        features = []
        labels = []
        coarse_labels = []
        for file in files:
            file_p = os.path.join(path, file)
            fea, lab, coarse_lab = self.read_file(file_p,
                                                  datakey=_CIFAR100_DATAKEY,
                                                  labelkey=[_CIFAR100_FINELABELKEY, _CIFAR100_COARSELABELKEY])
            features.append(fea)
            labels.append(lab)
            coarse_labels.append(coarse_lab)
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        coarse_labels = np.concatenate(coarse_labels)

        if self._one_hot:
            labels = transform.dense_to_one_hot(labels, self._cifar100_classes)
            coarse_labels = transform.dense_to_one_hot(coarse_labels, self._cifar100_super_classes)

        size = features.shape[0]

        features_test = features[:int(self._test_rate * size), :]
        features_train = features[int(self._test_rate * size):size, :]
        labels_test = labels[:int(self._test_rate * size), :]
        labels_train = labels[int(self._test_rate * size):size, :]
        coarse_labels_test = coarse_labels[:int(self._test_rate * size), :]
        coarse_labels_train = coarse_labels[int(self._test_rate * size):size, :]

        if self._reshape:
            features_train = transform.reshape(features_train, width, height, channel)
            features_test = transform.reshape(features_test, width, height, channel)

        return features_train, labels_train, features_test, labels_test, coarse_labels_train, coarse_labels_test

    def resize_data(self, width, height, channel, width_new, height_new, new_path, source_path=None, data_size=1000,
                    cifar10=True):
        """resize cifar image and save as new dataset"""
        self._one_hot = False
        self._reshape = False
        if not source_path:
            source_path = self._path

        if cifar10:
            features_train, labels_train, features_test, labels_test = self.load_cifar10(width, height, channel, source_path)
        else:
            features_train, labels_train, features_test, labels_test, coarse_labels_train, coarse_labels_test = self.load_cifar100(width, height, channel, source_path)

        total_size = len(features_train)
        file_size = math.ceil(total_size/data_size)
        for _ in six.moves.range(file_size):
            print('resize at %d / %d' % (_+1, file_size))
            index = _*data_size
            offset = total_size
            if _ + 1 != file_size:
                offset = (_+1)*data_size
            resize_features = transform.resize(features_train[index:offset],
                                               width, height, channel, width_new, height_new)
            if cifar10:
                self.save_file(resize_features, labels_train[index:offset], None,
                               new_path+'-'+str(width_new)+'x'+str(height_new), 'batch_'+str(_))
            else:
                self.save_file(resize_features, labels_train[index:offset], coarse_labels_train[index:offset],
                               new_path+'-'+str(width_new)+'x'+str(height_new), 'batch_'+str(_))
