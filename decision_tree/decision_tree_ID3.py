#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
This is realization of decision tree
'''

from math import log
import sys

sys.path.append('../tools')
import data_tool

#entropy computation
def entropy(pi_data):
    res = 0.0
    for item in pi_data:
        if item > 0.0:
            res += (-item * log(item, 2.0))
    return res

#classified label entropy
def target_entropy(labels):
    kinds_dict = {}
    for item in labels:
        if kinds_dict.__contains__(item):
            kinds_dict[item] += 1.0
        else:
            kinds_dict[item] = 0.0
    total = len(labels)
    pi_data = []
    for item in kinds_dict.values():
        pi_data.append(item / total)
    return entropy(pi_data)

#other properties entropy
def property_entropy(sub_data):
    data, total = data_tool.refer_split_count(sub_data)
    entropy = 0.0
    for item in data.items():
        each_item_sum = 0.0
        each_item_sub_sum = []
        sub_kind_sub_ratio = []
        for sub_item in item[-1]:
            each_item_sum += sub_item.values()[0]
            each_item_sub_sum.append(sub_item.values()[0])
        for count in each_item_sub_sum:
            sub_kind_sub_ratio.append(count/each_item_sum)
        entropy += (entropy(sub_kind_sub_ratio) * each_item_sum / total)
    print entropy

def info_gain(property_entropy, target_entropy):
    return target_entropy - property_entropy

def train(data, labels, max_depth):
    if len(data[0]) < max_depth:
        raise Exception('invalid depth')
    labels_entropy = target_entropy(labels)
    property_data = {}
    for row_index in range(len(data)):
        for column_index in range(len(data[row_index])):
            if not property_data.__contains__(column_index):
                property_data[column_index] = []
            temp_item = [data[row_index][column_index], labels[row_index]]
            property_data[column_index].append(temp_item)
    print 1

def predict(data):
    pred = []

def decision_tree(features_train, labels_train, features_test, labels_test, max_depth):
    print 1

sub_data=[]
sub_data.append(['s', '0'])
sub_data.append(['s', '1'])
sub_data.append(['l', '1'])
sub_data.append(['m', '1'])
sub_data.append(['l', '1'])
sub_data.append(['m', '0'])
sub_data.append(['m', '0'])
sub_data.append(['l', '0'])
sub_data.append(['m', '0'])
sub_data.append(['s', '1'])
labels=['0', '1', '1', '1', '1', '1', '0', '1', '1', '0']
# info_gain(sub_data)
train(sub_data, labels, 2)