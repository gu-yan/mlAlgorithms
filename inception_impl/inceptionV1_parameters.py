#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf


def init_param(number_classes=1000):
    model_param = {
        'conv1_kernel':
            tf.Variable(tf.truncated_normal(shape=[7, 7, 3, 64], stddev=1e-4, dtype=tf.float32)),
            'conv1_strides': 2, 'conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
            'conv1_name': 'conv1', 'conv1_padding': 'SAME',
        'max1_ksize': 3,
            'max1_strides': 2, 'max1_name': 'max1', 'max1_padding': 'SAME',
        'norm1_depth_radius': 2.5,
            'norm1_bias': 2.0, 'norm1_alpha': 1e-4, 'norm1_beta': 0.75, 'norm1_name': 'local_resp_norm1',
        'conv2_kernel': tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 192], stddev=1e-4, dtype=tf.float32)),
            'conv2_strides': 1, 'conv2_biases': tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32)),
            'conv2_name': 'conv2', 'conv2_padding': 'VALID',
        'conv3_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 192], stddev=1e-4, dtype=tf.float32)),
            'conv3_strides': 1, 'conv3_biases': tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32)),
            'conv3_name': 'conv3', 'conv3_padding': 'SAME',
        'norm2_depth_radius': 2.5,
            'norm2_bias': 2.0, 'norm2_alpha': 1e-4, 'norm2_beta': 0.75, 'norm2_name': 'local_resp_norm2',
        'inception1_name': 'inception1',
            'inception1_concat_axis': 3,
            'inception1_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception1_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 192, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception1_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 192, 96], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception1_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 96, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception1_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 192, 16], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception1_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception1_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception1_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 192, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception1_patch4_conv2', 'patch4_conv1_padding': 'SAME'
             },
        'inception2_name': 'inception2',
            'inception2_concat_axis': 3,
            'inception2_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception2_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception2_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception2_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception2_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception2_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 96], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception2_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception2_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception2_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'inception3_name': 'inception3',
            'inception3_concat_axis': 3,
            'inception3_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception3_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 480, 192], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception3_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 480, 96], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception3_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 96, 208], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[208], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception3_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 480, 16], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception3_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 48], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception3_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception3_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 480, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception3_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'inception4_name': 'inception4',
            'inception4_concat_axis': 3,
            'inception4_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception4_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 160], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[160], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception4_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 112], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[112], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception4_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 112, 224], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[224], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception4_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 24], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[24], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception4_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 24, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception4_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception4_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception4_patch4_conv2', 'patch4_conv1_padding': 'SAME',
                'patch5_avage1_kernel': 5,
                    'patch5_avage1_strides': 3, 'patch5_avage1_name': 'inception4_patch5_avage1',
                    'patch5_avage1_padding': 'VALID',
                'patch5_conv_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 328], stddev=1e-4, dtype=tf.float32)),
                    'patch5_conv_strides': 1,
                    'patch5_conv_biases': tf.Variable(tf.constant(0.0, shape=[328], dtype=tf.float32)),
                    'patch5_conv_name': 'inception4_patch5_conv1', 'patch5_conv_padding': 'SAME',
                'patch5_fc1_weights':
                    tf.Variable(tf.truncated_normal(shape=[328, 1024], stddev=1e-1, dtype=tf.float32)),
                    'patch5_fc1_biases': tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32)),
                    'patch5_fc1_name': 'inception4_patch5_fc1', 'patch5_fc1_relu': True,
                    'patch5_fc1_relu_bias': tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32)),
                'patch5_fc2_weights':
                    tf.Variable(tf.truncated_normal(shape=[1024, number_classes], stddev=1e-1, dtype=tf.float32)),
                    'patch5_fc2_biases': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32)),
                    'patch5_fc2_name': 'inception4_patch5_fc2', 'patch5_fc2_relu': True,
                    'patch5_fc2_relu_bias': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32))
            },
        'inception5_name': 'inception5',
            'inception5_concat_axis': 3,
            'inception5_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception5_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception5_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception5_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception5_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 24], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[24], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception5_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 24, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception5_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception5_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception5_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'inception6_name': 'inception6',
            'inception6_concat_axis': 3,
            'inception6_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception6_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 112], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[112], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception6_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 144], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[144], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception6_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 144, 288], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[288], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception6_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception6_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception6_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception6_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 64], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception6_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'inception7_name': 'inception7',
            'inception7_concat_axis': 3,
            'inception7_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception7_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 528, 256], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception7_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 528, 160], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[160], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception7_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 160, 320], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[320], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception7_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 528, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception7_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception7_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception7_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 528, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception7_patch4_conv2', 'patch4_conv1_padding': 'SAME',
                'patch5_avage1_kernel': 5,
                    'patch5_avage1_strides': 3, 'patch5_avage1_name': 'inception7_patch5_avage1',
                    'patch5_avage1_padding': 'VALID',
                'patch5_conv_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 528, 328], stddev=1e-4, dtype=tf.float32)),
                    'patch5_conv_strides': 1,
                    'patch5_conv_biases': tf.Variable(tf.constant(0.0, shape=[328], dtype=tf.float32)),
                    'patch5_conv_name': 'inception7_patch5_conv1', 'patch5_conv_padding': 'SAME',
                'patch5_fc1_weights':
                    tf.Variable(tf.truncated_normal(shape=[328, 1024], stddev=1e-1, dtype=tf.float32)),
                    'patch5_fc1_biases': tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32)),
                    'patch5_fc1_name': 'inception7_patch5_fc1', 'patch5_fc1_relu': True,
                    'patch5_fc1_relu_bias': tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32)),
                'patch5_fc2_weights':
                    tf.Variable(tf.truncated_normal(shape=[1024, number_classes], stddev=1e-1, dtype=tf.float32)),
                    'patch5_fc2_biases': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32)),
                    'patch5_fc2_name': 'inception7_patch5_fc2', 'patch5_fc2_relu': True,
                    'patch5_fc2_relu_bias': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32))
            },
        'inception8_name': 'inception8',
            'inception8_concat_axis': 3,
            'inception8_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception8_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 256], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception8_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 160], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[160], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception8_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 160, 320], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[320], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception8_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 32], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception8_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception8_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception8_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception8_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'inception9_name': 'inception9',
            'inception9_concat_axis': 3,
            'inception9_param': {
                'max1_ksize': 3, 'max1_strides': 2, 'max1_name': 'inception9_max1', 'max1_padding': 'SAME',
                'patch1_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 384], stddev=1e-4, dtype=tf.float32)),
                    'patch1_conv1_strides': 1,
                    'patch1_conv1_biases': tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32)),
                    'patch1_conv1_name': 'inception9_patch1_conv1', 'patch1_conv1_padding': 'SAME',
                'patch2_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 192], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv1_strides': 1,
                    'patch2_conv1_biases': tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32)),
                    'patch2_conv1_name': 'inception9_patch2_conv1', 'patch2_conv1_padding': 'SAME',
                'patch2_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 384], stddev=1e-4, dtype=tf.float32)),
                    'patch2_conv2_strides': 1,
                    'patch2_conv2_biases': tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32)),
                    'patch2_conv2_name': 'inception9_patch2_conv2', 'patch2_conv2_padding': 'SAME',
                'patch3_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 48], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv1_strides': 1,
                    'patch3_conv1_biases': tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32)),
                    'patch3_conv1_name': 'inception9_patch3_conv1', 'patch3_conv1_padding': 'SAME',
                'patch3_conv2_kernel':
                    tf.Variable(tf.truncated_normal(shape=[5, 5, 48, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch3_conv2_strides': 1,
                    'patch3_conv2_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch3_conv2_name': 'inception9_patch3_conv2', 'patch3_conv2_padding': 'SAME',
                'patch4_max1_kernel': 3,
                    'patch4_max1_strides': 1, 'patch4_max1_name': 'inception9_patch4_max1',
                    'patch4_max1_padding': 'SAME',
                'patch4_conv1_kernel':
                    tf.Variable(tf.truncated_normal(shape=[1, 1, 832, 128], stddev=1e-4, dtype=tf.float32)),
                    'patch4_conv1_strides': 1,
                    'patch4_conv1_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)),
                    'patch4_conv1_name': 'inception9_patch4_conv2', 'patch4_conv1_padding': 'SAME'
            },
        'avage1_ksize': 7, 'avage1_strides': 1, 'avage1_name': 'avage1', 'avage1_padding': 'SAME',
        'fc1_weights':  tf.Variable(tf.truncated_normal(shape=[1024, number_classes], stddev=1e-1, dtype=tf.float32)),
            'fc1_biases': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32)),
            'fc1_name': 'fc1', 'fc1_relu': True,
            'fc1_relu_bias': tf.Variable(tf.constant(1.0, shape=[number_classes], dtype=tf.float32))

    }
    return model_param
