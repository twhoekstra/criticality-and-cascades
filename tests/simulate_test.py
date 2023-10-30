import unittest
from utils.simulate import RecurrentNeuralNetwork

import matplotlib.pyplot as plt

class RecurrentNeuralNetworkTestCase(unittest.TestCase):

    def test_create(self):
        rnn = RecurrentNeuralNetwork(n=10)

        self.assertTrue(True)

    def test_sim_run(self):
        rnn = RecurrentNeuralNetwork(n=10)
        rnn.sim(100)

        self.assertTrue(True)

    def test_sim_plot_state(self):
        rnn = RecurrentNeuralNetwork(n=10)
        rnn.sim(100)
        fig, axs = rnn.plot_state(show=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)

    def test_sim_plot_connectivity(self):
        rnn = RecurrentNeuralNetwork(n=10, seed=1)
        rnn.sim(100)
        fig, axs = rnn.plot_connectivity(show=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)

    def test_sim_plot_state_big(self):
        rnn = RecurrentNeuralNetwork(n=10, g=10)
        rnn.sim(100)
        fig, axs = rnn.plot_state(show=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)


if __name__ == '__main__':
    unittest.main()
