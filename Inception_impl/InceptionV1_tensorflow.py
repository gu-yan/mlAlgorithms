#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
from tools import visualize


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


def local_response_normalization(data, name, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(input=data, depth_radius=depth_radius, bias=bias, alpha=alpha,
                                                  beta=beta, name=name)


def inception_module(data, model_param, concat_axis=2, train_flag=False):
    patch1 = convolution(data=data, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch2_1 = convolution(data=data, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch2_2 = convolution(data=patch2_1, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch3_1 = convolution(data=data, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch3_2 = convolution(data=patch3_1, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch4_1 = pooling(data=data, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    patch4_2 = convolution(data=patch4_1, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    inception = tf.concat(axis=concat_axis, values=[patch1, patch2_2, patch3_2, patch4_2])
    if train_flag:
        patch4_avage_pool = pooling(data=data)
        patch4_conv = convolution(data=patch4_avage_pool)
        patch4_fc1 = fullconnection(data=patch4_conv)
        patch4_2 = fullconnection(data=patch4_fc1)
        softmax = tf.nn.softmax(logits=, labels=)
        return inception, softmax
    return inception, None


def model(input, model_param):
    conv1 = convolution(data=input, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    max1 = pooling(data=conv1, ksize=model_param['pool1_ksize'], strides=model_param['pool1_strides'], name=model_param['pool1_name'], padding=model_param['pool1_padding'])
    local_res_norm1 = local_response_normalization(data=max1, depth_radius=model_param['norm1_depth_radius'], bias=model_param['norm1_bias'], alpha=model_param['norm1_alpha'], beta=model_param['norm1_beta'], name=model_param['norm1_name'])
    conv2 = convolution(data=local_res_norm1, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    conv3 = convolution(data=conv2, kernel=model_param['conv1_kernel'], strides=model_param['conv1_strides'], bias=model_param['conv1_biases'], name=model_param['conv1_name'], padding=model_param['conv1_padding'])
    local_res_norm2 = local_response_normalization(data=conv3, depth_radius=model_param['norm1_depth_radius'], bias=model_param['norm1_bias'], alpha=model_param['norm1_alpha'], beta=model_param['norm1_beta'], name=model_param['norm1_name'])
    max1 = pooling(data=local_res_norm2, ksize=model_param['pool1_ksize'], strides=model_param['pool1_strides'], name=model_param['pool1_name'], padding=model_param['pool1_padding'])
    inception1 = inception_module(data=max1, model_param=model_param['inception1_param'])
    inception2 = inception_module(data=inception1, model_param=model_param['inception2_param'])
    inception3 = inception_module(data=inception2, model_param=model_param['inception3_param'])
    inception4, softmax0 = inception_module(data=inception3, model_param=model_param['inception4_param'], train_flag=model_param['train_flag'])
    inception5 = inception_module(data=inception4, model_param=model_param['inception5_param'])
    inception6 = inception_module(data=inception5, model_param=model_param['inception6_param'])
    inception7, softmax1 = inception_module(data=inception6, model_param=model_param['inception7_param'], train_flag=model_param['train_flag'])
    inception8 = inception_module(data=inception7, model_param=model_param['inception8_param'])
    inception9 = inception_module(data=inception8, model_param=model_param['inception9_param'])
    avage_pool1 = pooling(data=inception9)
    return avage_pool1


def init_param():
    model_param = {

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
