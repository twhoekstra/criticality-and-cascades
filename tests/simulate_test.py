import unittest
from utils.simulate import RecurrentNeuralNetwork

import matplotlib.pyplot as plt

class RecurrentNeuralNetworkTestCase(unittest.TestCase):

    def test_create(self):
        rnn = RecurrentNeuralNetwork(n=10)

    def test_sim_run(self):
        rnn = RecurrentNeuralNetwork(n=10)
        rnn.sim(1200)

    def test_sim_plot_state(self):
        rnn = RecurrentNeuralNetwork(n=10)
        rnn.sim(1200)
        rnn.plot_state()

    def test_sim_plot_connectivity(self):
        rnn = RecurrentNeuralNetwork(n=100)
        rnn.sim(10)
        rnn.plot_connectivity()

    def test_sim_plot_state_big(self):
        rnn = RecurrentNeuralNetwork(n=100, g=10)
        rnn.sim(1200)
        rnn.plot_state()


if __name__ == '__main__':
    unittest.main()
