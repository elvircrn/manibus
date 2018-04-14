import tensorflow as tf
from tensorflow.contrib import slim

import data


def mini_vgg(inputs, is_training, dropout, scope=data.DEFAULT_SCOPE):
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=tf.nn.relu):
            # -4
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout1')

            # -6
            net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
            # / 2
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout2')

            # # -6
            # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
            # net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
            # # / 2
            # net = slim.dropout(net, keep_prob=dropout, is_training=is_training, scope='dropout3')

            net = slim.flatten(net)
            net = slim.fully_connected(net, data.N_CLASSES, activation_fn=None, scope='fc1')

            net = slim.softmax(net, scope='sm1')

        return net
