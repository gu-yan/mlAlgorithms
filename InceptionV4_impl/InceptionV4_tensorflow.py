#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf


def convolution(data, kernel, strides, name, bias, padding='SAME'):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input=data, filter=kernel, strides=[1, strides, strides, 1], padding=padding, name=name)
        with tf.name_scope(name+'_relu'):
            conv = tf.nn.bias_add(conv, bias)
            relu = tf.nn.relu(conv, name=name+'_relu')
            return relu


def pooling(data, ksize, strides, name, padding='SAME'):
    with tf.name_scope(name):
        return tf.nn.max_pool(value=data, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1],
                              padding=padding, name=name)


def fullconnection(data, weights, biases, name, relu_bias, relu=False):
    with tf.name_scope(name):
        fc = tf.nn.bias_add(tf.matmul(data, weights), biases, name=name)
        if not relu:
            return fc
        with tf.name_scope(name+'_relu'):
            fc = tf.nn.bias_add(fc, relu_bias)
            relu = tf.nn.relu(fc, name=(name+'_relu'))
            return relu