#!/usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import random

def load_data():
    with open('dataset.pkl', 'r') as file:
        data_set = pickle.load(file)
        return data_set

def feature_format(data_set):
    features = []
    labels = []
    for item in data_set:
        features.append(item[:-1])
        labels.append(item[-1])
    return features, labels

def train_test_split(features, test_rate):
    random.shuffle(features)
    total_number = len(features)
    test_number = int(round(len(features) * test_rate))
    train_data = features[0:-test_number]
    test_data = features[-test_number:total_number]
    features_train, labels_train = feature_format(train_data)
    features_test, labels_test = feature_format(test_data)
    return features_train, labels_train, features_test, labels_test
