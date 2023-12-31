import unittest

import numpy as np

from utils.simulate import RecurrentNeuralNetwork

import matplotlib.pyplot as plt

# plt.style.use('dark_background')

class RecurrentNeuralNetworkTestCase(unittest.TestCase):
    test_rnn = RecurrentNeuralNetwork(n=10,
                                      timestep_ms=0.1,
                                      seed=1)
    test_rnn.store('test')

    def test_create(self):
        rnn = RecurrentNeuralNetwork(n=10)

        self.assertTrue(True)

    def test_store(self):
        rnn = RecurrentNeuralNetwork(n=10)

        rnn.store()

        self.assertTrue(True)

    def test_restore(self):
        self.test_rnn.restore('test')
        self.assertTrue(True)

    def test_sim_run(self):
        self.test_rnn.restore('test')
        self.test_rnn.sim(100)

        self.assertTrue(True)

    def test_sim_run_activity_shape(self):
        self.test_rnn.restore('test')
        self.test_rnn.sim(100)
        v = self.test_rnn._monitors.v

        self.assertEquals(v.shape[1], 1000)
        self.assertEquals(v.shape[0], 10)

    def test_sim_run_leader_neuron(self):
        # No connections, seed in which leader neuron spikes
        rnn = RecurrentNeuralNetwork(n=10, p_c=0, seed=1)
        rnn.sim(1000)
        v = rnn.state_results.v_mV

        est_leader_neuron_idx = np.argmax(np.std(v, axis=1))

        self.assertEquals(est_leader_neuron_idx, rnn.leader_neuron_idx)

    def test_sim_run_nonzero(self):
        rnn = RecurrentNeuralNetwork(n=10, p_c=1, gamma=0.1, seed=1)
        rnn.sim(100)
        v = rnn.state_results.v_mV

        rnn.plot_voltage(show=True)

        self.assertTrue(np.nonzero(v))

    def test_sim_plot_state(self):
        self.test_rnn.restore('test')

        self.test_rnn.sim(1000)
        fig, axs = self.test_rnn.plot_voltage(show=True)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)

    def test_sim_various_w(self):
        ws = np.arange(9, 11, 0.1)

        rnn = RecurrentNeuralNetwork(n=1000, w=10.6)

        rnn.store('various_w')

        rnn.sim(1000)
        rnn.plot_voltage(show=True)
        # rnn.plot_connectivity(show=True)
        rnn.plot_spike_distribution(show=True)
        rnn.plot_spikes()

        # for i, w in enumerate(ws):
        #     rnn.restore('various_w')
        #
        #     rnn.set_w(w)
        #
        #     rnn.sim(1000)
        #
        #     # = rnn.plot_state(show=True)
        #     fig, axs = rnn.plot_spikes(show=False)
        #     axs[0].set_title(f'W={w:.2f}')
        #     fig.show()
        #     fig.savefig(f'{i:03}')
        #
        #
        # self.assertTrue(True)

    def test_sim_spike_distr(self):
        rnn = RecurrentNeuralNetwork(n=1000, w=10)
        rnn.sim(500)

        rnn.plot_spikes()
        rnn.plot_spike_distribution()

        self.assertTrue(True)

    def test_sim_plot_connectivity(self):
        rnn = RecurrentNeuralNetwork(n=10, p_c=0.5)
        fig, axs = rnn.plot_connectivity(show=True)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)

    def test_sim_plot_spikes_and_rates(self):
        self.test_rnn.restore('test')
        self.test_rnn.sim(1000)
        fig, axs = self.test_rnn.plot_spikes(show=True)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)

    def test_sim_plot_state_big(self):
        rnn = RecurrentNeuralNetwork(n=100, seed=1)
        rnn.sim(1000)
        fig, axs = rnn.plot_voltage(show=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(axs)


if __name__ == '__main__':
    unittest.main()
