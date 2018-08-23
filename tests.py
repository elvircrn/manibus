import unittest
import numpy as np
import tensorflow as tf

import network


class TestLoss(unittest.TestCase):
    def test_loss(self):
        labels = tf.constant([[[[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]]], dtype=tf.float64)
        predictions = tf.constant([[[[2, 3, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]]]], dtype=tf.float64)
        loss = network.calculate_loss(labels, predictions)
        self.assertAlmostEqual(loss.numpy(), 5 + 2 * np.power(np.sqrt(2) - 1, 2) + 2)

        print("loss: {}", loss)


if __name__ == '__main__':
    tf.enable_eager_execution()
    unittest.main()
