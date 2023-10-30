import unittest

import numpy as np

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

    def test_sim_run_activity_shape(self):
        rnn = RecurrentNeuralNetwork(n=10, timestep_ms=0.1)
        rnn.sim(100)
        v = rnn._state_monitor.v

        self.assertEquals(v.shape[1], 1000)
        self.assertEquals(v.shape[0], 10)

    def test_sim_run_leader_neuron(self):
        # No connections, seed in which leader neuron spikes
        rnn = RecurrentNeuralNetwork(n=10, p_c=0, seed=1)
        rnn.sim(100)
        v = rnn.state_results.v_mV

        est_leader_neuron_idx = np.argmax(np.std(v, axis=1))

        self.assertEquals(est_leader_neuron_idx, rnn.leader_neuron_idx)

    def test_sim_run_nonzero(self):
        rnn = RecurrentNeuralNetwork(n=10, p_c=1, gamma=0.1, seed=1)
        rnn.sim(100)
        v = rnn.state_results.v_mV

        rnn.plot_state(show=False)

        self.assertTrue(np.nonzero(v))

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
        rnn = RecurrentNeuralNetwork(n=100, seed=1)
        rnn.sim(1000)
        fig, axs = rnn.plot_state(show=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)


if __name__ == '__main__':
    unittest.main()
