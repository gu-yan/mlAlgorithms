#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
from tools import visualize
import inceptionV1_parameters


NUMBER_CLASSES = 10
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
DATASET_PATH = '/home/workspace/cifar10/pythonver'
LEARNING_RATE = 0.1
batch_size = 100
LOG_DIR = 'Log--' + visualize.get_time()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('train_mode', True, 'run in train mode or eval modo.')


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


def local_response_normalization(data, name, depth_radius=2.5, bias=2.0, alpha=1e-4, beta=0.75):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(input=data, depth_radius=depth_radius, bias=bias, alpha=alpha,
                                                  beta=beta, name=name)


def inception_module(data, model_param, name, concat_axis=3, train_flag=False, labels=None):
    with tf.name_scope(name):
        max1 = pooling(data=data,
                       ksize=model_param['max1_ksize'],
                       strides=model_param['max1_strides'],
                       name=model_param['max1_name'],
                       padding=model_param['max1_padding'])
        with tf.name_scope(name + 'patch1'):
            patch1 = convolution(data=max1,
                                 kernel=model_param['patch1_conv1_kernel'],
                                 strides=model_param['patch1_conv1_strides'],
                                 bias=model_param['patch1_conv1_biases'],
                                 name=model_param['patch1_conv1_name'],
                                 padding=model_param['patch1_conv1_padding'])
        with tf.name_scope(name + 'patch2'):
            patch2_1 = convolution(data=max1,
                                   kernel=model_param['patch2_conv1_kernel'],
                                   strides=model_param['patch2_conv1_strides'],
                                   bias=model_param['patch2_conv1_biases'],
                                   name=model_param['patch2_conv1_name'],
                                   padding=model_param['patch2_conv1_padding'])
            patch2_2 = convolution(data=patch2_1,
                                   kernel=model_param['patch2_conv2_kernel'],
                                   strides=model_param['patch2_conv2_strides'],
                                   bias=model_param['patch2_conv2_biases'],
                                   name=model_param['patch2_conv2_name'],
                                   padding=model_param['patch2_conv2_padding'])
        with tf.name_scope(name + 'patch3'):
            patch3_1 = convolution(data=max1,
                                   kernel=model_param['patch3_conv1_kernel'],
                                   strides=model_param['patch3_conv1_strides'],
                                   bias=model_param['patch3_conv1_biases'],
                                   name=model_param['patch3_conv1_name'],
                                   padding=model_param['patch3_conv1_padding'])
            patch3_2 = convolution(data=patch3_1,
                                   kernel=model_param['patch3_conv2_kernel'],
                                   strides=model_param['patch3_conv2_strides'],
                                   bias=model_param['patch3_conv2_biases'],
                                   name=model_param['patch3_conv2_name'],
                                   padding=model_param['patch3_conv2_padding'])
        with tf.name_scope(name + 'patch4'):
            patch4_1 = pooling(data=max1,
                               ksize=model_param['patch4_max1_kernel'],
                               strides=model_param['patch4_max1_strides'],
                               name=model_param['patch4_max1_name'],
                               padding=model_param['patch4_max1_padding'])
            patch4_2 = convolution(data=patch4_1,
                                   kernel=model_param['patch4_conv1_kernel'],
                                   strides=model_param['patch4_conv1_strides'],
                                   bias=model_param['patch4_conv1_biases'],
                                   name=model_param['patch4_conv1_name'],
                                   padding=model_param['patch4_conv1_padding'])
        inception = tf.concat(axis=concat_axis, values=[patch1, patch2_2, patch3_2, patch4_2])
        if train_flag:
            with tf.name_scope(name + 'patch5'):
                patch5_avage_pool = pooling(data=max1,
                                            ksize=model_param['patch5_avage1_kernel'],
                                            strides=model_param['patch5_avage1_strides'],
                                            name=model_param['patch5_avage1_name'],
                                            padding=model_param['patch5_avage1_padding'])
                patch5_conv = convolution(data=patch5_avage_pool,
                                          kernel=model_param['patch5_conv_kernel'],
                                          strides=model_param['patch5_conv_strides'],
                                          bias=model_param['patch5_conv_biases'],
                                          name=model_param['patch5_conv_name'],
                                          padding=model_param['patch5_conv_padding'])
                patch5_conv = tf.reshape(patch5_conv,
                                         shape=[-1, model_param['patch5_fc1_weights'].get_shape()[0].value])
                patch5_fc1 = fullconnection(data=patch5_conv,
                                            weights=model_param['patch5_fc1_weights'],
                                            biases=model_param['patch5_fc1_biases'],
                                            name=model_param['patch5_fc1_name'],
                                            relu_bias=model_param['patch5_fc1_relu_bias'],
                                            relu=model_param['patch5_fc1_relu'])
                patch5_fc2 = fullconnection(data=patch5_fc1,
                                            weights=model_param['patch5_fc2_weights'],
                                            biases=model_param['patch5_fc2_biases'],
                                            name=model_param['patch5_fc2_name'],
                                            relu_bias=model_param['patch5_fc2_relu_bias'],
                                            relu=model_param['patch5_fc2_relu'])
                softmax = tf.nn.softmax_cross_entropy_with_logits(logits=patch5_fc2, labels=labels)
            return inception, softmax
    return inception, None


def model(input, labels, model_param, train_mode=True):
    conv1 = convolution(data=input,
                        kernel=model_param['conv1_kernel'],
                        strides=model_param['conv1_strides'],
                        bias=model_param['conv1_biases'],
                        name=model_param['conv1_name'],
                        padding=model_param['conv1_padding'])
    max1 = pooling(data=conv1,
                   ksize=model_param['max1_ksize'],
                   strides=model_param['max1_strides'],
                   name=model_param['max1_name'],
                   padding=model_param['max1_padding'])
    local_resp_norm1 = local_response_normalization(data=max1,
                                                    depth_radius=model_param['norm1_depth_radius'],
                                                    bias=model_param['norm1_bias'],
                                                    alpha=model_param['norm1_alpha'],
                                                    beta=model_param['norm1_beta'],
                                                    name=model_param['norm1_name'])
    conv2 = convolution(data=local_resp_norm1,
                        kernel=model_param['conv2_kernel'],
                        strides=model_param['conv2_strides'],
                        bias=model_param['conv2_biases'],
                        name=model_param['conv2_name'],
                        padding=model_param['conv2_padding'])
    conv3 = convolution(data=conv2,
                        kernel=model_param['conv3_kernel'],
                        strides=model_param['conv3_strides'],
                        bias=model_param['conv3_biases'],
                        name=model_param['conv3_name'],
                        padding=model_param['conv3_padding'])
    local_resp_norm2 = local_response_normalization(data=conv3,
                                                    depth_radius=model_param['norm2_depth_radius'],
                                                    bias=model_param['norm2_bias'],
                                                    alpha=model_param['norm2_alpha'],
                                                    beta=model_param['norm2_beta'],
                                                    name=model_param['norm2_name'])
    inception1 = inception_module(data=local_resp_norm2,
                                  name=model_param['inception1_name'],
                                  model_param=model_param['inception1_param'],
                                  concat_axis=model_param['inception1_concat_axis'])
    inception2 = inception_module(data=inception1,
                                  name=model_param['inception2_name'],
                                  model_param=model_param['inception2_param'],
                                  concat_axis=model_param['inception2_concat_axis'])
    inception3 = inception_module(data=inception2,
                                  name=model_param['inception3_name'],
                                  model_param=model_param['inception3_param'],
                                  concat_axis=model_param['inception3_concat_axis'])
    inception4, softmax0 = inception_module(data=inception3,
                                            name=model_param['inception4_name'],
                                            model_param=model_param['inception4_param'],
                                            concat_axis=model_param['inception4_concat_axis'],
                                            train_flag=train_mode,
                                            labels=labels)
    inception5 = inception_module(data=inception4,
                                  name=model_param['inception5_name'],
                                  model_param=model_param['inception5_param'],
                                  concat_axis=model_param['inception5_concat_axis'])
    inception6 = inception_module(data=inception5,
                                  name=model_param['inception6_name'],
                                  model_param=model_param['inception6_param'],
                                  concat_axis=model_param['inception6_concat_axis'])
    inception7, softmax1 = inception_module(data=inception6,
                                            name=model_param['inception7_name'],
                                            model_param=model_param['inception7_param'],
                                            concat_axis=model_param['inception7_concat_axis'],
                                            train_flag=train_mode,
                                            labels=labels)
    inception8 = inception_module(data=inception7,
                                  name=model_param['inception8_name'],
                                  model_param=model_param['inception8_param'],
                                  concat_axis=model_param['inception8_concat_axis'])
    inception9 = inception_module(data=inception8,
                                  name=model_param['inception9_name'],
                                  model_param=model_param['inception9_param'],
                                  concat_axis=model_param['inception9_concat_axis'])
    avage_pool1 = pooling(data=inception9,
                          ksize=model_param['avage1_ksize'],
                          strides=model_param['avage1_strides'],
                          name=model_param['avage1_name'],
                          padding=model_param['avage1_padding'])
    fc1 = fullconnection(data=avage_pool1,
                         weights=model_param['fc1_weights'],
                         biases=model_param['fc1_biases'],
                         name=model_param['fc1_name'],
                         relu_bias=model_param['fc1_relu_bias'],
                         relu=model_param['fc1_relu'])
    return fc1, softmax0, softmax1


def train_act(features_train, labels_train, features_test, labels_test):
    sess = tf.InteractiveSession()

    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], name='features')
    y = tf.placeholder(dtype=tf.float32, shape=[None, NUMBER_CLASSES], name='labels')
    model_paramters = inceptionV1_parameters.init_param()
    pred, softmax0, softmax1 = model(input=x, labels=y, model_param=model_paramters, train_mode=FLAGS.train_mode)
    with tf.name_scope('final_loss'):
        final_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('final_loss', final_loss)
    tf.summary.histogram('final_loss', final_loss)
    with tf.name_scope('final_train'):
        final_train = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(final_loss)

    if FLAGS.train_mode:
        with tf.name_scope('loss0'):
            loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar('loss0', loss0)
        tf.summary.histogram('loss0', loss0)
        with tf.name_scope('train0'):
            train0 = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss0)

        with tf.name_scope('loss1'):
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar('loss1', loss1)
        tf.summary.histogram('loss1', loss1)
        with tf.name_scope('train1'):
            train1 = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss1)

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
        if FLAGS.train_mode:
            sess.run([final_train, train0, train1], feed_dict={x: data, y: labels})
        else:
            sess.run(final_train, feed_dict={x: data, y: labels})
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
