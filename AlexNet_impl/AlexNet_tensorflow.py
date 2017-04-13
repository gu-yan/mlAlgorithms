#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tools import visualize

#TODO visualize each layer


NUMBER_CLASSES = 10
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
DATASET_PATH = '/home/workspace/cifar10/pythonver'
LEARNING_RATE = 0.1
DROPOUT_PROB1 = 0.5
DROPOUT_PROB2 = 0.5
batch_size = 100
LOG_DIR = 'Log--' + visualize.get_time()


def convolution(data, kernel, strides, name, padding='SAME'):
    with tf.name_scope(name):
        return tf.nn.conv2d(input=data, filter=kernel, strides=[1, strides, strides, 1], padding=padding, name=name)


def pooling(data, ksize, strides, name, padding='SAME', type='max'):
    with tf.name_scope(name):
        if type == 'max':
            return tf.nn.max_pool(value=data, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1],
                                  padding=padding, name=name)
        return tf.nn.avg_pool(value=data, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding,
                              name=name)


def fullconnection(data, weights, biases, name, relu=False):
    with tf.name_scope(name):
        fc = tf.nn.bias_add(tf.matmul(data, weights), biases, name=name)
    if not relu:
        return fc
    with tf.name_scope(name+'_relu'):
        relu = tf.nn.relu(fc, name=(name+'_relu'))
        return relu


def local_response_normalization(data, name, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(input=data, depth_radius=depth_radius, bias=bias, alpha=alpha,
                                                  beta=beta, name=name)


def dropout(data, name, prob=0.5):
    with tf.name_scope(name):
        return tf.nn.dropout(x=data, keep_prob=prob)


def model(input, model_param):
    conv1 = convolution(data=input,
                        kernel=model_param['conv1_kernel'],
                        strides=model_param['conv1_strides'],
                        name=model_param['conv1_name'],
                        padding=model_param['conv1_padding'])
    local_res_norm1 = local_response_normalization(data=conv1,
                                                   depth_radius=model_param['norm1_depth_radius'],
                                                   bias=model_param['norm1_bias'],
                                                   alpha=model_param['norm1_alpha'],
                                                   beta=model_param['norm1_beta'],
                                                   name=model_param['norm1_name'])
    max_pool1 = pooling(data=local_res_norm1,
                        ksize=model_param['pool1_ksize'],
                        strides=model_param['pool1_strides'],
                        name=model_param['pool1_name'],
                        padding=model_param['pool1_padding'],
                        type=model_param['pool1_type'])
    conv2 = convolution(data=max_pool1,
                        kernel=model_param['conv2_kernel'],
                        strides=model_param['conv2_strides'],
                        name=model_param['conv2_name'],
                        padding=model_param['conv2_padding'])
    local_res_norm2 = local_response_normalization(data=conv2,
                                                   depth_radius=model_param['norm2_depth_radius'],
                                                   bias=model_param['norm2_bias'],
                                                   alpha=model_param['norm2_alpha'],
                                                   beta=model_param['norm2_beta'],
                                                   name=model_param['norm2_name'])
    max_pool2 = pooling(data=local_res_norm2,
                        ksize=model_param['pool2_ksize'],
                        strides=model_param['pool2_strides'],
                        name=model_param['pool2_name'],
                        padding=model_param['pool2_padding'],
                        type=model_param['pool2_type'])
    conv3 = convolution(data=max_pool2,
                        kernel=model_param['conv3_kernel'],
                        strides=model_param['conv3_strides'],
                        name=model_param['conv3_name'],
                        padding=model_param['conv3_padding'])
    conv4 = convolution(data=conv3,
                        kernel=model_param['conv4_kernel'],
                        strides=model_param['conv4_strides'],
                        name=model_param['conv4_name'],
                        padding=model_param['conv4_padding'])
    conv5 = convolution(data=conv4,
                        kernel=model_param['conv5_kernel'],
                        strides=model_param['conv5_strides'],
                        name=model_param['conv5_name'],
                        padding=model_param['conv5_padding'])
    max_pool3 = pooling(data=conv5,
                        ksize=model_param['pool3_ksize'],
                        strides=model_param['pool3_strides'],
                        name=model_param['pool3_name'],
                        padding=model_param['pool3_padding'],
                        type=model_param['pool3_type'])
    max_pool3 = tf.reshape(max_pool3, shape=[-1, model_param['fc1_weights'].get_shape()[0].value])
    fc1 = fullconnection(data=max_pool3,
                         weights=model_param['fc1_weights'],
                         biases=model_param['fc1_biases'],
                         name=model_param['fc1_name'],
                         relu=model_param['fc1_relu'])
    dropout1 = dropout(data=fc1,
                       prob=model_param['dropout1_prob'],
                       name=model_param['dropout1_name'])
    fc2 = fullconnection(data=dropout1,
                         weights=model_param['fc2_weights'],
                         biases=model_param['fc2_biases'],
                         name=model_param['fc2_name'],
                         relu=model_param['fc2_relu'])
    dropout2 = dropout(data=fc2,
                       prob=model_param['dropout2_prob'],
                       name=model_param['dropout2_name'])
    fc3 = fullconnection(data=dropout2,
                         weights=model_param['fc3_weights'],
                         biases=model_param['fc3_biases'],
                         name=model_param['fc3_name'],
                         relu=model_param['fc3_relu'])
    return fc3


def init_param():
    model_param = {
        'conv1_kernel': tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 64], stddev=1e-2, dtype=tf.float32)),
                      'conv1_strides': 4, 'conv1_name': 'conv1', 'conv1_padding': 'SAME',
        'norm1_depth_radius': 2,
                      'norm1_bias': 1.0, 'norm1_alpha': 2e-05, 'norm1_beta': 0.75, 'norm1_name': 'local_res_norm1',
        'pool1_ksize': 3,
                      'pool1_strides': 2, 'pool1_name': 'maxpool1', 'pool1_padding': 'SAME', 'pool1_type': 'max',
        'conv2_kernel': tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 192], stddev=1e-2, dtype=tf.float32)),
                      'conv2_strides': 1, 'conv2_name': 'conv2', 'conv2_padding': 'SAME',
        'norm2_depth_radius': 2,
                      'norm2_bias': 1.0, 'norm2_alpha': 2e-05, 'norm2_beta': 0.75, 'norm2_name': 'local_res_norm2',
        'pool2_ksize': 3,
                      'pool2_strides': 2, 'pool2_name': 'maxpool2', 'pool2_padding': 'SAME', 'pool2_type': 'max',
        'conv3_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 384], stddev=1e-2, dtype=tf.float32)),
                      'conv3_strides': 1, 'conv3_name': 'conv3', 'conv3_padding': 'SAME',
        'conv4_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256], stddev=1e-2, dtype=tf.float32)),
                      'conv4_strides': 1, 'conv4_name': 'conv4', 'conv4_padding': 'SAME',
        'conv5_kernel': tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=1e-2, dtype=tf.float32)),
                      'conv5_strides': 1, 'conv5_name': 'conv5', 'conv5_padding': 'SAME',
        'pool3_ksize': 3,
                      'pool3_strides': 2, 'pool3_name': 'maxpool3', 'pool3_padding': 'SAME', 'pool3_type': 'max',
        'fc1_weights': tf.Variable(tf.truncated_normal(shape=[256, 4096], stddev=1e-2, dtype=tf.float32)),
                      'fc1_biases': tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32)),
                      'fc1_name': 'full_conn_1', 'fc1_relu': True,
        'dropout1_prob': DROPOUT_PROB1,
                      'dropout1_name': 'dropout1',
        'fc2_weights': tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=1e-2, dtype=tf.float32)),
                      'fc2_biases': tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32)),
                      'fc2_name': 'full_conn_2', 'fc2_relu': True,
        'dropout2_prob': DROPOUT_PROB2,
                      'dropout2_name': 'dropout2',
        'fc3_weights': tf.Variable(tf.truncated_normal(shape=[4096, NUMBER_CLASSES], stddev=1e-2, dtype=tf.float32)),
                      'fc3_biases': tf.Variable(tf.constant(0.0, shape=[NUMBER_CLASSES], dtype=tf.float32)),
                      'fc3_name': 'full_conn_3', 'fc3_relu': True,
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
        train = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
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
            print(visualize.get_time() + '   train_iteration at %d, test score: %f ' % (i, accuracy_res))
    sess.close()
    logwriter.close()


def main():
    from tools import dataset
    features_train, labels_train, features_test, labels_test = dataset.load_cifar10(DATASET_PATH,
                                                                                    one_hot=True,
                                                                                    num_classes=NUMBER_CLASSES,
                                                                                    test_rate=0.3)
    train_act(features_train.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]), labels_train,
              features_test.reshape([-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]), labels_test)


if __name__ == '__main__':
    main()
