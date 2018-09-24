import tensorflow as tf
from tensorflow.contrib import slim

import data


def hands_mini_vgg(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # -4 -> 24x24
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
            # / 2 -> 12x12
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')

            # -6 -> 6x6
            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
            # / 2 -> 3x3
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.HANDS_N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')
        return net


def yolo_fast(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # 640x360
            predictions = slim.conv2d(inputs, 6, [640 - 16, 360 - 9], padding='VALID', scope='conv1')
            return predictions


def yolo_arch_fast(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            net = slim.repeat(inputs, 2, slim.conv2d, 1, [3, 3], padding='SAME', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')
            # 320x180

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 3], padding='SAME', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')
            # 160x90

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 3], padding='SAME', scope='conv3')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout3')
            # 80x45

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 3], padding='SAME', scope='conv4')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout4')
            # 40x22

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 3], padding='SAME', scope='conv5')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool5')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout5')
            # 20x11

            net = slim.conv2d(net, 6, [4, 2], padding='VALID', scope='conv6')
            # net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout6')
            # 17x10

            # net = slim.softmax(net, scope='sm1')

    return net


def yolo_arch_fast_020(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # 256x144
            net = slim.repeat(inputs, 1, slim.conv2d, 1, [3, 3], padding='VALID', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')
            # 128x72

            net = slim.repeat(net, 1, slim.conv2d, 1, [3, 3], padding='VALID', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 1, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')
            # 64x36

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 3], padding='VALID', scope='conv3')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout3')
            # 32x18

            net = slim.repeat(net, 3, slim.conv2d, 1, [3, 2], padding='VALID', scope='conv4')
            # net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout4')

            net = slim.conv2d(net, 6, [6, 2], padding='VALID', scope='conv5')
            # net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout5')

            net = slim.softmax(net, scope='sm1')

    return net


def yolo_arch_slow_020(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # 256x144
            net = slim.repeat(inputs, 1, slim.conv2d, 32, [3, 3], padding='VALID', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')
            # 128x72

            net = slim.repeat(net, 1, slim.conv2d, 32, [3, 3], padding='VALID', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 1, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')
            # 64x36

            net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], padding='VALID', scope='conv3')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout3')
            # 32x18

            net = slim.repeat(net, 3, slim.conv2d, 64, [3, 2], padding='VALID', scope='conv4')
            # net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout4')

            net = slim.conv2d(net, 6, [6, 2], padding='VALID', scope='conv5')
            # net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout5')

            net = slim.softmax(net, scope='sm1')

    return net


def yolo_arch_slow(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')
            # 320x180

            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')
            # 160x90

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout3')
            # 80x45

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv4')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout4')
            # 40x22

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv5')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool5')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout5')
            # 20x11

            net = slim.conv2d(net, 6, [4, 1], padding='VALID', scope='conv6')
            # net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout6')
            # 17x10

            # net = slim.softmax(net, scope='sm1')

    return net
