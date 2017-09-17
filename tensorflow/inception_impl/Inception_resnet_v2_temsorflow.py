#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import six
from tools import visualize


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('cls', 10, 'numbers of classifiers')
tf.app.flags.DEFINE_integer('IMG_WIDTH', 299, 'image width')
tf.app.flags.DEFINE_integer('IMG_HEIGHT', 299, 'image height')
tf.app.flags.DEFINE_integer('IMG_CHANNEL', 3, 'image channel')
tf.app.flags.DEFINE_integer('batch_size', 5, 'batch size')
tf.app.flags.DEFINE_integer('epoches', 1, 'epoches')
tf.app.flags.DEFINE_float('LEARNING_RATE', 0.0001, 'learning rate for momentum GD')
tf.app.flags.DEFINE_float('MOMENTUM', 0.9, 'momentum rate for momentum GD')
tf.app.flags.DEFINE_string('data_path', '/home/workspace/cc', 'path of dataset')
tf.app.flags.DEFINE_string('LOG_DIR', 'Log--' + visualize.get_time(), 'tensorboard log dir')


def convolution(data, kernel, strides, padding='SAME', name=None):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input=data,
                            filter=tf.Variable(tf.truncated_normal(shape=kernel, stddev=1e-2, dtype=tf.float32)),
                            strides=[1, strides, strides, 1], padding=padding, name=name)
        return conv


def pooling(data, ksize, strides, padding='SAME', type='max', name=None):
    with tf.name_scope(name):
        if type == 'max':
            pool = tf.nn.max_pool(value=data, ksize=[1, ksize, ksize, 1],
                                  strides=[1, strides, strides, 1], padding=padding, name=name)
        else:
            pool = tf.nn.avg_pool(value=data, ksize=[1, ksize, ksize, 1],
                                  strides=[1, strides, strides, 1], padding=padding, name=name)
        return pool


def relu(data, name=None):
    with tf.name_scope(name):
        relu_res = tf.nn.relu(data, name=name)
        return relu_res


def fullconnection(data, weights, biases, name):
    with tf.name_scope(name):
        fc = tf.nn.xw_plus_b(data, weights, biases, name=name)
        return fc


def stem(data):
    with tf.name_scope('stem'):
        conv1 = convolution(data, kernel=[3, 3, FLAGS.IMG_CHANNEL, 32], strides=2, padding='VALID',
                            name='stem_conv1')
        conv2 = convolution(conv1, kernel=[3, 3, 32, 32], strides=1, padding='VALID', name='stem_conv2')
        conv3 = convolution(conv2, kernel=[3, 3, 32, 64], strides=1, name='stem_conv3')

        # block1
        block1_max_pool = pooling(conv3, ksize=3, strides=2, padding='VALID', name='stem_block1_max_pool')
        block1_conv = convolution(conv3, kernel=[3, 3, 64, 96], strides=2, padding='VALID', name='stem_block1_conv')
        block1 = tf.concat(axis=3, values=[block1_max_pool, block1_conv])

        # block2_1
        block2_1_conv1 = convolution(block1, kernel=[1, 1, 160, 64], strides=1, name='block2_1_conv1')
        block2_1_conv2 = convolution(block2_1_conv1, kernel=[3, 3, 64, 96], strides=1, padding='VALID',
                                     name='block2_1_conv2')

        # block2_2
        block2_2_conv1 = convolution(block1, kernel=[1, 1, 160, 64], strides=1, name='block2_2_conv1')
        block2_2_conv2 = convolution(block2_2_conv1, kernel=[7, 1, 64, 64], strides=1, name='block2_2_conv2')
        block2_2_conv3 = convolution(block2_2_conv2, kernel=[1, 7, 64, 64], strides=1, name='block2_2_conv3')
        block2_2_conv4 = convolution(block2_2_conv3, kernel=[3, 3, 64, 96], strides=1, padding='VALID',
                                     name='block2_2_conv4')
        block2 = tf.concat(axis=3, values=[block2_1_conv2, block2_2_conv4])

        # block3
        block3_conv = convolution(block2, kernel=[3, 3, 192, 192], strides=2, padding='VALID', name='block3_conv')
        block3_pool = pooling(block2, ksize=3, strides=2, padding='VALID', name='block3_pool')
        block3 = tf.concat(axis=3, values=[block3_conv, block3_pool])

        return block3


def inception_resnet_a(data):
    with tf.name_scope('inception_resnet_a'):
        relu1 = relu(data, name='relu1')

        # patch1
        patch1_conv1 = convolution(data, kernel=[1, 1, 384, 32], strides=1, name='patch1_conv1')

        # patch2
        patch2_conv1 = convolution(data, kernel=[1, 1, 384, 32], strides=1, name='patch2_conv1')
        patch2_conv2 = convolution(patch2_conv1, kernel=[3, 3, 32, 32], strides=1, name='patch2_conv2')

        # patch3
        patch3_conv1 = convolution(data, kernel=[1, 1, 384, 32], strides=1, name='patch3_conv1')
        patch3_conv2 = convolution(patch3_conv1, kernel=[3, 3, 32, 48], strides=1, name='patch3_conv2')
        patch3_conv3 = convolution(patch3_conv2, kernel=[3, 3, 48, 64], strides=1, name='patch3_conv3')

        patch = tf.concat(axis=3, values=[patch1_conv1, patch2_conv2, patch3_conv3])
        patch_conv = convolution(patch, kernel=[1, 1, 128, 384], strides=1, name='patch_conv')

        relu1 += patch_conv
        return relu1


def reduction_a(data):
    with tf.name_scope('reduction_a'):
        patch1 = pooling(data, ksize=3, strides=2, padding='VALID', name='patch1')
        patch2 = convolution(data, kernel=[3, 3, 384, 384], strides=2, padding='VALID', name='patch2')
        patch3_conv1 = convolution(data, kernel=[1, 1, 384, 256], strides=1, name='patch3_conv1')
        patch3_conv2 = convolution(patch3_conv1, kernel=[3, 3, 256, 256], strides=1, name='patch3_conv2')
        patch3_conv3 = convolution(patch3_conv2, kernel=[3, 3, 256, 384], strides=2, padding='VALID',
                                   name='patch3_conv3')
        patch = tf.concat(axis=3, values=[patch1, patch2, patch3_conv3])
        return patch


def inception_resnet_b(data):
    with tf.name_scope('inception_resnet_b'):
        relu1 = relu(data, name='relu1')

        # patch1
        patch1_conv1 = convolution(data, kernel=[1, 1, 1152, 192], strides=1, name='patch1_conv1')

        # patch2
        patch2_conv1 = convolution(data, kernel=[1, 1, 1152, 128], strides=1, name='patch2_conv1')
        patch2_conv2 = convolution(patch2_conv1, kernel=[1, 7, 128, 160], strides=1, name='patch2_conv2')
        patch2_conv3 = convolution(patch2_conv2, kernel=[7, 1, 160, 192], strides=1, name='patch2_conv3')

        patch = tf.concat(axis=3, values=[patch1_conv1, patch2_conv3])
        patch = convolution(patch, kernel=[1, 1, 384, 1152], strides=1, name='patch_conv')

        relu1 += patch
        return relu1


def reduction_b(data):
    # patch1
    patch1 = pooling(data, ksize=3, strides=2, padding='VALID', name='patch1_max')

    # patch2
    patch2_conv1 = convolution(data, kernel=[1, 1, 1152, 256],  strides=1, name='patch2_conv1')
    patch2_conv2 = convolution(patch2_conv1, kernel=[3, 3, 256, 384], strides=2, padding='VALID', name='patch2_conv2')

    # patch3
    patch3_conv1 = convolution(data, kernel=[1, 1, 1152, 256], strides=1, name='patch3_conv1')
    patch3_conv2 = convolution(patch3_conv1, kernel=[3, 3, 256, 288], strides=2, padding='VALID', name='patch3_conv2')

    # patch4
    patch4_conv1 = convolution(data, kernel=[1, 1, 1152, 256], strides=1, name='patch4_conv1')
    patch4_conv2 = convolution(patch4_conv1, kernel=[3, 3, 256, 288], strides=1, name='patch4_conv2')
    patch4_conv3 = convolution(patch4_conv2, kernel=[3, 3, 288, 320], strides=2, padding='VALID', name='patch4_conv3')

    patch = tf.concat(axis=3, values=[patch1, patch2_conv2, patch3_conv2, patch4_conv3])
    return patch


def inception_resnet_c(data):
    with tf.name_scope('inception_resnet_c'):
        relu1 = relu(data, name='relu1')

        # patch1
        patch1_conv1 = convolution(data, kernel=[1, 1, 2144, 192], strides=1, name='patch1_conv1')

        # patch2
        patch2_conv1 = convolution(data, kernel=[1, 1, 2144, 192], strides=1, name='patch2_conv1')
        patch2_conv2 = convolution(patch2_conv1, kernel=[1, 3, 192, 224], strides=1, name='patch2_conv2')
        patch2_conv3 = convolution(patch2_conv2, kernel=[3, 1, 224, 256], strides=1, name='patch2_conv3')

        patch = tf.concat(axis=3, values=[patch1_conv1, patch2_conv3])
        patch = convolution(patch, kernel=[1, 1, 448, 2144], strides=1, name='patch')

        relu1 += patch
        return relu1


def model(data):
    stem_unit = stem(data)
    inception_resnet_a_unit = stem_unit
    for _ in six.moves.range(5):
        inception_resnet_a_unit = inception_resnet_a(inception_resnet_a_unit)
    reduction_a_unit = reduction_a(inception_resnet_a_unit)
    inception_resnet_b_unit = reduction_a_unit
    for _ in six.moves.range(10):
        inception_resnet_b_unit = inception_resnet_b(inception_resnet_b_unit)
    reduction_b_unit = reduction_b(inception_resnet_b_unit)
    inceptio_resnet_c_unit = reduction_b_unit
    for _ in six.moves.range(5):
        inceptio_resnet_c_unit = inception_resnet_c(inceptio_resnet_c_unit)
    avg_pool_unit = pooling(inceptio_resnet_c_unit, ksize=8, strides=1, padding='VALID', type='avg', name='avg_pool')
    drop = tf.nn.dropout(avg_pool_unit, keep_prob=0.8)
    drop = tf.reshape(drop, shape=[-1, 2144])
    fc = fullconnection(drop,
                        weights=tf.Variable(tf.truncated_normal(shape=[2144, FLAGS.cls], dtype=tf.float32)),
                        biases=tf.Variable(tf.truncated_normal(shape=[FLAGS.cls], dtype=tf.float32)), name='full_conn')
    return fc


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
    iterations = int(data_size / FLAGS.batch_size)
    for _ in range(FLAGS.epoches):
        for i in range(iterations):
            data = []
            labels = []
            if i == iterations - 1:
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
    from tools import dataset
    features_train, labels_train, features_test, labels_test = dataset.load_cifar10(FLAGS.data_path,
                                                                                    width=FLAGS.IMG_WIDTH,
                                                                                    height=FLAGS.IMG_HEIGHT,
                                                                                    one_hot=True)
    train_act(features_train, labels_train, features_test, labels_test)


if __name__ == '__main__':
    main()
