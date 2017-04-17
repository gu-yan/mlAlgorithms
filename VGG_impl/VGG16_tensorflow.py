#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tools import visualize


NUMBER_CLASSES = 10
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
DATASET_PATH = '/home/workspace/cifar10/pythonver'
LEARNING_RATE = 0.001
MOMENTUM = 0.9
batch_size = 128
epoches = 5
LOG_DIR = 'Log--' + visualize.get_time()


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


def model(input, model_param):
    conv1 = convolution(data=input, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    conv2 = convolution(data=conv1, kernel=model_param['conv2_kernel'], strides=model_param['conv2_strides'], bias=model_param['conv2_biases'], name=model_param['conv2_name'], padding=model_param['conv2_padding'])

    pool1 = pooling(data=conv2, ksize=model_param['pool1_ksize'], strides=model_param['pool1_strides'], name=model_param['pool1_name'], padding=model_param['pool1_padding'])

    conv3 = convolution(data=pool1, kernel=model_param['conv3_kernel'], strides=model_param['conv3_strides'], bias=model_param['conv3_biases'], name=model_param['conv3_name'], padding=model_param['conv3_padding'])
    conv4 = convolution(data=conv3, kernel=model_param['conv4_kernel'], strides=model_param['conv4_strides'], bias=model_param['conv4_biases'], name=model_param['conv4_name'], padding=model_param['conv4_padding'])

    pool2 = pooling(data=conv4, ksize=model_param['pool2_ksize'], strides=model_param['pool2_strides'], name=model_param['pool2_name'], padding=model_param['pool2_padding'])

    conv5 = convolution(data=pool2, kernel=model_param['conv5_kernel'], strides=model_param['conv5_strides'], bias=model_param['conv5_biases'], name=model_param['conv5_name'], padding=model_param['conv5_padding'])
    conv6 = convolution(data=conv5, kernel=model_param['conv6_kernel'], strides=model_param['conv6_strides'], bias=model_param['conv6_biases'], name=model_param['conv6_name'], padding=model_param['conv6_padding'])
    conv7 = convolution(data=conv6, kernel=model_param['conv7_kernel'], strides=model_param['conv7_strides'], bias=model_param['conv7_biases'], name=model_param['conv7_name'], padding=model_param['conv7_padding'])

    pool3 = pooling(data=conv7, ksize=model_param['pool3_ksize'], strides=model_param['pool3_strides'], name=model_param['pool3_name'], padding=model_param['pool3_padding'])

    conv8 = convolution(data=pool3, kernel=model_param['conv8_kernel'], strides=model_param['conv8_strides'], bias=model_param['conv8_biases'], name=model_param['conv8_name'], padding=model_param['conv8_padding'])
    conv9 = convolution(data=conv8, kernel=model_param['conv9_kernel'], strides=model_param['conv9_strides'], bias=model_param['conv9_biases'], name=model_param['conv9_name'], padding=model_param['conv9_padding'])
    conv10 = convolution(data=conv9, kernel=model_param['conv10_kernel'], strides=model_param['conv10_strides'], bias=model_param['conv10_biases'], name=model_param['conv10_name'], padding=model_param['conv10_padding'])

    pool4 = pooling(data=conv10, ksize=model_param['pool4_ksize'], strides=model_param['pool4_strides'], name=model_param['pool4_name'], padding=model_param['pool4_padding'])

    conv11 = convolution(data=pool4, kernel=model_param['conv11_kernel'], strides=model_param['conv11_strides'], bias=model_param['conv11_biases'], name=model_param['conv11_name'], padding=model_param['conv11_padding'])
    conv12 = convolution(data=conv11, kernel=model_param['conv12_kernel'], strides=model_param['conv12_strides'], bias=model_param['conv12_biases'], name=model_param['conv12_name'], padding=model_param['conv12_padding'])
    conv13 = convolution(data=conv12, kernel=model_param['conv13_kernel'], strides=model_param['conv13_strides'], bias=model_param['conv13_biases'], name=model_param['conv13_name'], padding=model_param['conv13_padding'])

    pool5 = pooling(data=conv13, ksize=model_param['pool5_ksize'], strides=model_param['pool5_strides'], name=model_param['pool5_name'], padding=model_param['pool5_padding'])
    pool5 = tf.reshape(pool5, shape=[-1, model_param['fc1_weights'].get_shape()[0].value])

    fc1 = fullconnection(data=pool5, weights=model_param['fc1_weights'], biases=model_param['fc1_biases'], relu_bias=model_param['fc1_relu_biases'], name=model_param['fc1_name'], relu=model_param['fc1_relu'])
    fc2 = fullconnection(data=fc1, weights=model_param['fc2_weights'], biases=model_param['fc2_biases'], relu_bias=model_param['fc2_relu_biases'], name=model_param['fc2_name'], relu=model_param['fc2_relu'])
    fc3 = fullconnection(data=fc2, weights=model_param['fc3_weights'], biases=model_param['fc3_biases'], relu_bias=model_param['fc3_relu_biases'], name=model_param['fc3_name'])
    return fc3


def init_param():
    model_param = {
        'conv1_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1e-2, dtype=tf.float32)), 'conv1_strides': 1, 'conv1_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)), 'conv1_name': 'conv1', 'conv1_padding': 'SAME',
        'conv2_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=1e-2, dtype=tf.float32)), 'conv2_strides': 1, 'conv2_biases': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32)), 'conv2_name': 'conv2', 'conv2_padding': 'SAME',
        'pool1_ksize': 2, 'pool1_strides': 2, 'pool1_name': 'pool1', 'pool1_padding': 'SAME',
        'conv3_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=1e-2, dtype=tf.float32)), 'conv3_strides': 1, 'conv3_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)), 'conv3_name': 'conv3', 'conv3_padding': 'SAME',
        'conv4_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=1e-2, dtype=tf.float32)), 'conv4_strides': 1, 'conv4_biases': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32)), 'conv4_name': 'conv4', 'conv4_padding': 'SAME',
        'pool2_ksize': 2, 'pool2_strides': 2, 'pool2_name': 'pool2', 'pool2_padding': 'SAME',
        'conv5_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=1e-2, dtype=tf.float32)), 'conv5_strides': 1, 'conv5_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)), 'conv5_name': 'conv5', 'conv5_padding': 'SAME',
        'conv6_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=1e-2, dtype=tf.float32)), 'conv6_strides': 1, 'conv6_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)), 'conv6_name': 'conv6', 'conv6_padding': 'SAME',
        'conv7_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=1e-2, dtype=tf.float32)), 'conv7_strides': 1, 'conv7_biases': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32)), 'conv7_name': 'conv7', 'conv7_padding': 'SAME',
        'pool3_ksize': 2, 'pool3_strides': 2, 'pool3_name': 'pool3', 'pool3_padding': 'SAME',
        'conv8_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=1e-2, dtype=tf.float32)), 'conv8_strides': 1, 'conv8_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv8_name': 'conv8', 'conv8_padding': 'SAME',
        'conv9_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1e-2, dtype=tf.float32)), 'conv9_strides': 1, 'conv9_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv9_name': 'conv9', 'conv9_padding': 'SAME',
        'conv10_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1e-2, dtype=tf.float32)), 'conv10_strides': 1, 'conv10_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv10_name': 'conv10', 'conv10_padding': 'SAME',
        'pool4_ksize': 2, 'pool4_strides': 2, 'pool4_name': 'pool4', 'pool4_padding': 'SAME',
        'conv11_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1e-2, dtype=tf.float32)), 'conv11_strides': 1, 'conv11_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv11_name': 'conv11', 'conv11_padding': 'SAME',
        'conv12_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1e-2, dtype=tf.float32)), 'conv12_strides': 1, 'conv12_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv12_name': 'conv12', 'conv12_padding': 'SAME',
        'conv13_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1e-2, dtype=tf.float32)), 'conv13_strides': 1, 'conv13_biases': tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32)), 'conv13_name': 'conv13', 'conv13_padding': 'SAME',
        'pool5_ksize': 2, 'pool5_strides': 2, 'pool5_name': 'pool5', 'pool5_padding': 'SAME',
        'fc1_weights': tf.Variable(tf.truncated_normal(shape=[512, 4096], stddev=1e-1, dtype=tf.float32)), 'fc1_biases': tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32)), 'fc1_relu_biases': tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32)), 'fc1_name': 'fc1', 'fc1_relu': True,
        'fc2_weights': tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=1e-1, dtype=tf.float32)), 'fc2_biases': tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32)), 'fc2_relu_biases': tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32)), 'fc2_name': 'fc2', 'fc2_relu': True,
        'fc3_weights': tf.Variable(tf.truncated_normal(shape=[4096, NUMBER_CLASSES], stddev=1e-1, dtype=tf.float32)), 'fc3_biases': tf.Variable(tf.constant(1.0, shape=[NUMBER_CLASSES], dtype=tf.float32)), 'fc3_relu_biases': tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32)), 'fc3_name': 'fc3'
    }
    return model_param


def train_act(features_train, labels_train, features_test, labels_test):
    sess = tf.InteractiveSession()

    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], name='features')
    y = tf.placeholder(dtype=tf.float32, shape=[None, NUMBER_CLASSES], name='labels')
    pred = model(x, init_param())
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM).minimize(loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('accuracy', accuracy)

    merge = tf.summary.merge_all()
    logwriter = tf.summary.FileWriter(LOG_DIR, sess.graph)
    initial = tf.global_variables_initializer()

    sess.run(initial)
    data_size = features_train.shape[0]
    iterations = int(data_size/batch_size)
    for _ in range(epoches):
        for i in range(iterations):
            data = []
            labels = []
            if i == iterations-1:
                data = features_train[i * batch_size: data_size, :, :, :]
                labels = labels_train[i * batch_size: data_size]
            else:
                data = features_train[i * batch_size: (i + 1) * batch_size, :, :, :]
                labels = labels_train[i * batch_size: (i + 1) * batch_size]
            sess.run(train, feed_dict={x: data, y: labels})
            if i % 10 == 0:
                summary, accuracy_res = sess.run([merge, accuracy], feed_dict={x: features_test, y: labels_test})
                logwriter.add_summary(summary, i)
                print(visualize.get_time() + '   epoch %d, train_iteration at %d, test score: %f ' % (_, i, accuracy_res))
    sess.close()
    logwriter.close()


def main():
    import dataset
    features_train, labels_train, features_test, labels_test = dataset.load_cifar10(DATASET_PATH,
                                                                                    one_hot=True,
                                                                                    num_classes=NUMBER_CLASSES,
                                                                                    test_rate=0.0016)
    train_act(features_train.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]), labels_train,
              features_test.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]), labels_test)


if __name__ == '__main__':
    main()
