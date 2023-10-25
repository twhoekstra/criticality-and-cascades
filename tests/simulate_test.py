import unittest
from utils.simulate import RecurrentNeuralNetwork

import matplotlib.pyplot as plt


class RecurrentNeuralNetworkTestCase(unittest.TestCase):

    def test_create(self):
        rnn = RecurrentNeuralNetwork(n=10)

    def test_sim_run(self):
        rnn = RecurrentNeuralNetwork(n=10)
        rnn.sim(1200)


if __name__ == '__main__':
    unittest.main()
