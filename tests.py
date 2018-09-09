import unittest

import numpy as np
import tensorflow as tf
import architectures as arch

import network


class TestLoss(unittest.TestCase):
    def test_loss(self):
        labels = tf.constant([[[[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]]], dtype=tf.float32)
        predictions = tf.constant([[[[2, 3, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]]]], dtype=tf.float32)
        loss = network.calculate_loss(labels, predictions)
        self.assertAlmostEqual(loss.numpy(), 5 + 2 * np.power(np.sqrt(2) - 1, 2) + 2)

        print("loss: {}", loss)

    def test_arch(self):
        labels = tf.constant(np.random.rand(64, 640, 320, 1), dtype=tf.float32)
        predictions = arch.yolo_arch_fast(labels, False, 0.5)
        truth = arch.yolo_arch_fast(labels, False, 0.5)
        print(network.calculate_loss(truth, predictions).numpy())
        print(network.calculate_loss(truth, predictions).numpy().shape)



if __name__ == '__main__':
    tf.enable_eager_execution()
    unittest.main()
