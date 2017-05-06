#!/usr/bin/python
# -*- coding:utf-8 -*-

from tools import cifar_dataset


def load_cifar10(file_path, width=32, height=32, one_hot=False, test_rate=0.3):
    cfd = cifar_dataset.CifarDataset(path=file_path, test_rate=test_rate, one_hot=one_hot, reshape=True)
    return cfd.load_cifar10(width=width, height=height, channel=3)


def load_cifar100(file_path, width=32, height=32, one_hot=False, test_rate=0.3):
    cfd = cifar_dataset.CifarDataset(path=file_path, test_rate=test_rate, one_hot=one_hot, reshape=True)
    return cfd.load_cifar100(width=width, height=height, channel=3)