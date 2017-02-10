#!/usr/bin/python
# -*-coding:utf-8 -*-

from math import exp
import random
import data_tool

#y = x1*a1 + x2*a2 + x3*a3 + ... + xn*an + b
def predict(data,
            coef,
            bias):
    pred = 0.0
    for index in range(len(coef)):
        pred += (data[index] * coef[index] + bias)
    return sigmoid(pred)

def sigmoid(x):
    res = 0.0
    try :
        if x > 60:
            res = 1.0 / (1.0 + exp(-60))
        elif x < -60:
            res = 1.0 / (1.0 + exp(60))
        else:
            res = 1.0 / (1.0 + exp(-x))
    except:
        print 'over math.exp range ', x
    return res

def sgd(train,
        labels,
        coef,
        bias,
        learn_rate,
        nepoch):
    for epoch in range(nepoch):
        sum_error = 0.0
        for index in range(len(train)):
            pred = predict(train[index], coef, bias)
            sum_error += (labels[index] - pred)
            bias = (bias + learn_rate * sum_error * pred * (1 - pred))
            for i in range(len(coef)):
                coef[i] = (coef[i] + learn_rate * sum_error * pred * (1 - pred) * train[index][i])
    return coef, bias

#generate standard normal distribution
def param_gauss(size):
    param = []
    for i in range(size):
        param.append(random.gauss(mu=0, sigma=0.05))
    return param

def logistic_regression(features_train, labels_train,
                        features_test, labels_test,
                        learn_rate, nepoch):
    coef = param_gauss(len(features_train[0]))
    bias = param_gauss(1)[0]
    coef, bias = sgd(features_train, labels_train, coef, bias, learn_rate, nepoch)
    pred = []
    for index in range(len(features_test)):
        pred.append(predict(features_test[index], coef, bias=bias))
    return pred, coef, bias

def accuracy(pred, y_true):
    correct = 0.0
    for index in range(len(pred)):
        if pred[index] == y_true[index]:
            correct += 1.0
    return correct / len(pred)



#test
features_train, labels_train, features_test, labels_test = data_tool.train_test_split(
    data_tool.load_data(),
    test_rate=0.3)

for i in range(5):
    print 'cycle +++++++++++++++++++++++++++++++++++++++++++++++++++++   ', i
    pred, coef, bias = logistic_regression(features_train, labels_train, features_test, labels_test,
        learn_rate=0.02, nepoch=100)
    score = accuracy(pred, labels_test)
    print 'coef is: ', coef
    print 'bias is: ', bias
    print 'accuracy is: ', score
