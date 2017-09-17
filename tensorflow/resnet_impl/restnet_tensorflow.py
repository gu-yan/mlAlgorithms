#!/usr/bin/python
# -*- coding:utf-8 -*-

import six
import tensorflow as tf
from tensorflow.python.training import moving_averages

from tools import visualize, dataset

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('train_mode', True, 'True--train, False--eval')
tf.app.flags.DEFINE_boolean('bottleneck_residual_flag', True, 'True--bottleneck residual, False--residual')
tf.app.flags.DEFINE_integer('residual_numbers', 50, 'numbers of residual module')
tf.app.flags.DEFINE_integer('cls', 10, 'numbers of classifiers')
tf.app.flags.DEFINE_integer('IMG_WIDTH', 32, 'image width')
tf.app.flags.DEFINE_integer('IMG_HEIGHT', 32, 'image height')
tf.app.flags.DEFINE_integer('IMG_CHANNEL', 3, 'image channel')
tf.app.flags.DEFINE_integer('batch_size', 12, 'batch size')
tf.app.flags.DEFINE_integer('epoches', 1, 'epoches')
tf.app.flags.DEFINE_float('LEARNING_RATE', 0.1, 'learning rate for momentum GD')
tf.app.flags.DEFINE_float('MOMENTUM', 0.9, 'momentum rate for momentum GD')
tf.app.flags.DEFINE_string('data_path', 'cifar10-pythonver', 'path of dataset')
tf.app.flags.DEFINE_string('LOG_DIR', 'Log--' + visualize.get_time(), 'tensorboard log dir')


def convolution(data, kernel, strides, name, padding='SAME'):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input=data, filter=kernel, strides=[1, strides, strides, 1], padding=padding, name=name)
        return conv


def pooling(data, ksize, strides, name, padding='VALID'):
    with tf.name_scope(name):
        pool = tf.nn.avg_pool(value=data, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1],
                              padding=padding, name=name)
        return pool


def leaky_relu(data, name, leak=0.0):
    with tf.name_scope(name):
        relu_res = tf.where(tf.less(data, 0.0), leak*data, data, name=name)
        return relu_res


def fullconnection(data, weights, biases, name):
    with tf.name_scope(name):
        fc = tf.nn.xw_plus_b(data, weights, biases, name=name)
        return fc


def batch_norm(data, name):
    shape_param = data.get_shape()[-1]
    beta = tf.get_variable(name=name+'_beta', shape=shape_param, dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name=name+'_gamma', shape=shape_param, dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    if FLAGS.train_mode:
        mean_param, variance_param = tf.nn.moments(x=data, axes=[0, 1, 2], name=name+'_moments')
        moving_mean = tf.get_variable(name=name+'_moving_mean', shape=shape_param, dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
        moving_variance = tf.get_variable(name=name+'_moving_variance', shape=shape_param, dtype=tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
        mean = moving_averages.assign_moving_average(variable=moving_mean, value=mean_param, decay=0.9)
        variance = moving_averages.assign_moving_average(variable=moving_variance, value=variance_param, decay=0.9)
    else:
        mean = tf.get_variable(name=name+'_moving_mean', shape=shape_param, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
        variance = tf.get_variable(name=name+'_moving_variance', shape=shape_param, dtype=tf.float32,
                                   initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
        tf.summary.scalar(mean.op.name, mean)
        tf.summary.scalar(variance.op.name, variance)
    b_norm = tf.nn.batch_normalization(x=data, mean=mean, variance=variance,
                                       offset=beta, scale=gamma, variance_epsilon=0.001, name=name)
    return b_norm


def bottleneck_residual_unit(data, name, in_filter, origin_data_flag=False):
    with tf.name_scope(name+'bottleneck_residual'):
        res_b_norm1 = batch_norm(data=data, name=name+'res_batch_norm1')
        res_relu1 = leaky_relu(data=res_b_norm1, name='res_relu1', leak=0.1)
        if origin_data_flag:
            data = res_relu1
        with tf.name_scope('sub_1'):
            res_conv1 = convolution(data=res_relu1,
                                    kernel=tf.Variable(tf.truncated_normal(
                                        shape=[1, 1, in_filter, 64], stddev=1e-2, dtype=tf.float32)),
                                    strides=1,
                                    name='res_conv1', padding='SAME')
        with tf.name_scope('sub_2'):
            res_b_norm2 = batch_norm(data=res_conv1, name=name+'res_batch_norm2')
            res_relu2 = leaky_relu(data=res_b_norm2, name='res_relu2', leak=0.1)
            res_conv2 = convolution(data=res_relu2,
                                    kernel=tf.Variable(tf.truncated_normal(
                                        shape=[3, 3, 64, 64], stddev=1e-2, dtype=tf.float32)),
                                    strides=1,
                                    name='res_conv2', padding='SAME')
        with tf.name_scope('sub_3'):
            res_b_norm3 = batch_norm(data=res_conv2, name=name+'res_batch_norm3')
            res_relu3 = leaky_relu(data=res_b_norm3, name='res_relu3', leak=0.1)
            res_conv3 = convolution(data=res_relu3,
                                    kernel=tf.Variable(tf.truncated_normal(
                                        shape=[1, 1, 64, 256], stddev=1e-2, dtype=tf.float32)),
                                    strides=1,
                                    name='res_conv3', padding='SAME')
        with tf.name_scope('add'):
            if data.get_shape()[-1] != res_conv3.get_shape()[-1]:
                data = convolution(data=data,
                                   kernel=tf.Variable(tf.truncated_normal(
                                       shape=[1, 1, in_filter, 256], stddev=1e-2, dtype=tf.float32)),
                                   strides=1,
                                   name='padding', padding='SAME')
            data += res_conv3
    return data


def residual_unit(data, name, in_filter, origin_data_flag=False):
    with tf.name_scope(name+'bottleneck_residual'):
        res_b_norm1 = batch_norm(data=data, name=name+'res_batch_norm1')
        res_relu1 = leaky_relu(data=res_b_norm1, name='res_relu1', leak=0.1)
        if origin_data_flag:
            data = res_relu1
        with tf.name_scope('sub_1'):
            res_conv1 = convolution(data=res_relu1, kernel=3, strides=1, name='res_conv1', padding='SAME')
        with tf.name_scope('sub_2'):
            res_b_norm2 = batch_norm(data=res_conv1, name=name+'res_batch_norm2')
            res_relu2 = leaky_relu(data=res_b_norm2, name='res_relu2', leak=0.1)
            res_conv2 = convolution(data=res_relu2, kernel=3, strides=1, name='res_conv2', padding='SAME')
        with tf.name_scope('add'):
            if data.get_shape()[-1] != res_conv2.get_shape()[-1]:
                data = convolution(data=data,
                                   kernel=tf.Variable(tf.truncated_normal(
                                       shape=[1, 1, in_filter, 256], stddev=1e-2, dtype=tf.float32)),
                                   strides=1,
                                   name='padding', padding='SAME')
            data += res_conv2
    return data


def model(data):
    for _ in six.moves.range(1, FLAGS.residual_numbers):
        origin_data_flag = False
        in_filter = 256
        if _ == 1:
            origin_data_flag = True
            in_filter = FLAGS.IMG_CHANNEL
        if FLAGS.bottleneck_residual_flag:
            residual_m = bottleneck_residual_unit(data=data,
                                                  name=str(_)+'b_r',
                                                  in_filter=in_filter, origin_data_flag=origin_data_flag)
        else:
            residual_m = residual_unit(data=data,
                                       name=str(_)+'b_r',
                                       in_filter=in_filter, origin_data_flag=origin_data_flag)
        data = residual_m

    data = batch_norm(data=data, name='last_batch_norm')
    data = leaky_relu(data=data, name='last_leaky_relu', leak=0.1)
    data = tf.reduce_mean(input_tensor=data, axis=[1, 2])
    data = tf.reshape(data, shape=[-1, 256])
    data = fullconnection(data=data,
                          weights=tf.Variable(tf.truncated_normal(shape=[256, FLAGS.cls], dtype=tf.float32)),
                          biases=tf.Variable(tf.constant(1.0, shape=[FLAGS.cls], dtype=tf.float32)), name='full_conn')
    return data


def train_act(features_train, labels_train, features_test, labels_test):
    sess = tf.InteractiveSession()
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None, FLAGS.IMG_WIDTH, FLAGS.IMG_HEIGHT, FLAGS.IMG_CHANNEL], name='features')
    y = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.cls], name='labels')
    pred = model(data=x)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss', loss)
    with tf.name_scope('train'):
        train = tf.train.MomentumOptimizer(learning_rate=FLAGS.LEARNING_RATE, momentum=FLAGS.MOMENTUM).minimize(loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('accuracy', accuracy)

    merge = tf.summary.merge_all()
    logwriter = tf.summary.FileWriter(FLAGS.LOG_DIR, sess.graph)
    initial = tf.global_variables_initializer()

    sess.run(initial)
    data_size = features_train.shape[0]
    iterations = int(data_size/FLAGS.batch_size)
    for _ in range(FLAGS.epoches):
        for i in range(iterations):
            data = []
            labels = []
            if i == iterations-1:
                data = features_train[i * FLAGS.batch_size: data_size, :, :, :]
                labels = labels_train[i * FLAGS.batch_size: data_size]
            else:
                data = features_train[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size, :, :, :]
                labels = labels_train[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            sess.run(train, feed_dict={x: data, y: labels})
            if i % 10 == 0:
                summary, accuracy_res = sess.run([merge, accuracy], feed_dict={x: features_test, y: labels_test})
                logwriter.add_summary(summary, i)
                print(visualize.get_time() +
                      '   epoch %d, train_iteration at %d, test score: %f ' % (_, i, accuracy_res))
    sess.close()
    logwriter.close()


def main():
    features_train, labels_train, features_test, labels_test = dataset.load_cifar10(FLAGS.data_path,
                                                                                    width=FLAGS.IMG_WIDTH,
                                                                                    height=FLAGS.IMG_HEIGHT,
                                                                                    one_hot=True)
    train_act(features_train, labels_train, features_test, labels_test)


if __name__ == '__main__':
    main()
