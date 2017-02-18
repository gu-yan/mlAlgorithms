#!/usr/bin/python
# -*- coding:utf-8 -*-

import pickle
import random

def load_data():
    with open('../tools/dataset.pkl', 'r') as file:
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


'''
split data, example:
        origin data:
          value     label
            s         0
            s         1
            m         1
            m         1
            m         0
        tidy data:
            {s: [s;0       1],
                [s;1       1]},
            {m: [m;1       2],
                [m;0       1]}
'''
def refer_split_count(data):
    res = {}
    dict_list = []

    total = float(len(data))

    # get labels, kinds of type
    label_enum = []
    kinds = []
    for item in data:
        label_enum.append(item[-1])
        kinds.append(item[0])
    label_enum = list(set(label_enum))
    kinds = list(set(kinds))
    for item in kinds:
        res[item] = []
    '''
    part1:
    after this part, the data becomes to:
            s;0       1
            s;1       1
            m;1       2
            m;0       1
    '''
    for item in data:
        sub_cls = item[0]
        label = item[-1]
        key = (str(sub_cls) + ';' + str(label))
        dict = {}
        dict[key] = 1.0
        update = False #identify if the list has containd the dict
        for index in range(len(dict_list)):
            temp_dict = dict_list[index]
            if dict.keys()[0] == temp_dict.keys()[0]:
                temp_dict[key] += 1.0
                dict_list[index] = temp_dict
                update = True
                break
        if not update:
            dict_list.append(dict)
            # add other label dicts
            for item in label_enum:
                if label != item:
                    dict_list.append({str(sub_cls) + ';' + str(item): 0.0})

    '''
    part2:
    parts append result from part1 to final dict
    '''
    for index in range(len(dict_list)):
        dict_temp = dict_list[index]
        kind = dict_temp.keys()[0].split(';')[0]
        res[kind].append(dict_temp)
    return res, total