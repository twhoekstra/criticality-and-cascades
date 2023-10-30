from dataclasses import dataclass
from typing import Optional

from brian2 import (defaultclock, ms, Hz, mV, NeuronGroup, Synapses,
                    StateMonitor, SpikeMonitor, PopulationRateMonitor,
                    PoissonInput, Network, collect)
import matplotlib.pyplot as plt
import numpy as np


class SimulationNotRunException(Exception):
    pass


class StateResults:
    """Results of StateMonitor"""

    def __init__(self, monitor: StateMonitor):
        self.v_mV: np.ndarray = monitor.v / mV
        self.t_ms: np.ndarray = monitor.t / ms

    def as_array(self):
        return np.vstack((self.t_ms, self.v_mV))


class SpikeResults:
    """Results of StateMonitor"""

    def __init__(self, monitor: SpikeMonitor):
        self.i: np.ndarray = monitor.i
        self.t_ms: np.ndarray = monitor.t / ms

    def as_array(self):
        return np.vstack((self.t_ms, self.i))


class RateResults:
    """Results of RateMonitor"""

    def __init__(self, monitor: PopulationRateMonitor):
        self.rate: np.ndarray = monitor.rate / Hz
        self.t_ms: np.ndarray = monitor.t / ms

    def as_array(self):
        return np.vstack((self.t_ms, self.rate))


class RecurrentNeuralNetwork:
    """Recurrent Neural Network of LIF neurons.

    Recurrent Neural Network (RNN) of Leaky-Integrate-and-Fire (LIF) neurons.
    Simulation based on Brunel, N. 2000 example of the ``brian2`` Python
    package: https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_2000.html#example-brunel-2000

    The recurrent neural network has excitatory in inhibitory neurons.
    No self connections

    Args:
        n: Number of neurons in the network.
        p_c: Connection probability. Defaults to 4% to make a sparse network.
        g: Relative strength of inhibitory synapses with respect to exitatory
          synapses.
        gamma: Fraction of neurons in the network that are excitatory.
        p_ext: Probability for random activity in a neuron per time step.
          Adds noise.

    Attributes:
        n: Number of neurons in the network.
        p_c: Connection probability.
        g: Relative strength of inhibitory synapses with respect to exitatory
          synapses.
        gamma: Fraction of neurons in the network that are excitatory.
        p_ext: Probability for random activity in a neuron per time step.
        _rate_monitor: Stores instantaneous spiking rates of groups of neurons.
        _spike_monitor: Stores spikes from individual neurons.
    """

    tau = 20 * ms
    theta = 20 * mV
    V_r = 10 * mV
    tau_rp = 2 * ms

    J = 0.1 * mV
    D = 1.5 * ms

    def __init__(self,
                 n: int = 128,
                 g: float = 3,
                 p_c: float = 0.04,
                 gamma=0.8,
                 p_ext: float = 6E-4,
                 nu_ext_over_nu_thr=0.9,
                 seed: int = None,
                 leader_freq: float = 10,
                 leader_mV: float = 10,
                 timestep_ms: float = 0.1):

        # Parameters
        self.n = n
        self.n_e = None
        self.p_c = p_c
        self.g = 3
        self.gamma = gamma
        self.p_ext = p_ext
        self.timestep_ms = timestep_ms
        self.leader_neuron_idx = None
        self.leader_rate = leader_freq
        self.leader_weight = leader_mV

        self.nu_ext_over_nu_thr = nu_ext_over_nu_thr

        # Brain2 Objects
        self.neurons = None
        self.synapses = None

        self._monitors = None

        self.net: Optional[Network] = None

        self.state_results: Optional[StateResults] = None
        self.spike_results: Optional[SpikeResults] = None
        self.rate_results: Optional[RateResults] = None

        # Set random number generator seed (optional)
        np.random.seed(seed)

    def sim(self, sim_time_ms):
        self._setup_network()

        tau = self.tau
        tau_rp = self.tau_rp
        V_r = self.V_r
        theta = self.theta

        g = self.g
        D = self.D
        J = self.J

        self.net.run(sim_time_ms * ms, report='text')

        self.rate_results = RateResults(self._rate_monitor)
        self.state_results = StateResults(self._state_monitor)
        self.spike_results = SpikeResults(self._spike_monitor)

        return 1

    def store(self):
        self.net.store()

    def restore(self):
        self.net.restore()

    def _setup_network(self):
        # network parameters
        C_E, self.n_e = self._get_network_params()

        # external stimulus
        nu_thr = self._get_external_stim_amp(C_E)

        # Noise rate
        rate = self._get_noise_rate()

        defaultclock.dt = self.timestep_ms * ms

        neurons = NeuronGroup(self.n,
                              """
                              dv/dt = -v/tau : volt (unless refractory)
                              """,
                              threshold="v > theta",
                              reset="v = V_r",
                              refractory=self.tau_rp,
                              method="exact",
                              )

        self.synapses = self._get_synapses(neurons, self.n_e)

        nu_ext = self.nu_ext_over_nu_thr * nu_thr
        external_poisson_input = PoissonInput(
            target=neurons, target_var="v", N=self.n, rate=rate,
            weight=self.J
        )

        idx = np.random.randint(0, self.n_e)
        self.leader_neuron_idx = idx
        leading_neuron = neurons[idx:idx + 1]

        leading_neuron_input = PoissonInput(
            target=leading_neuron,
            target_var="v", N=1, rate=self.leader_rate * Hz,
            weight=self.leader_weight * mV
        )

        self._monitors = self.setup_monitors(neurons)
        self.neurons = neurons

        self.net = Network(collect())
        self.net.add(self._monitors)
        self.net.add(self.synapses)

    def _get_synapses(self, neurons, N_E):

        excitatory_neurons = neurons[:N_E]
        inhibitory_neurons = neurons[N_E:]

        exc_synapses = Synapses(excitatory_neurons,
                                target=neurons,
                                on_pre="v += J",
                                delay=self.D,
                                name="exc")
        exc_synapses.connect(p=self.p_c, condition='i!=j')
        inhib_synapses = Synapses(inhibitory_neurons,
                                  target=neurons,
                                  on_pre="v += -g*J",
                                  delay=self.D,
                                  name="inhib")
        inhib_synapses.connect(p=self.p_c,
                               condition='i!=j')  # No self connections

        return exc_synapses, inhib_synapses

    def _get_external_stim_amp(self, C_E):
        return self.theta / (self.J * C_E * self.tau)

    def _get_noise_rate(self):
        rate = self.p_ext / (self.timestep_ms * ms)
        return rate

    def _get_network_params(self):
        N_E = round(self.gamma * self.n)

        epsilon = 0.1
        C_E = epsilon * N_E

        return C_E, N_E

    def setup_monitors(self, neurons, i=50):
        self._rate_monitor = PopulationRateMonitor(neurons)
        # record from the first i excitatory neurons
        self._spike_monitor = SpikeMonitor(neurons[:i])
        self._state_monitor = StateMonitor(neurons[:i], 'v', record=True)

        return self._rate_monitor, self._spike_monitor, self._state_monitor

    def plot_connectivity(self, show=True):
        if self.exc_synapses is None or self.inhib_synapses is None:
            raise SimulationNotRunException('Error, no results yet. '
                                            'Did you run the simulation?')

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(np.zeros(self.n), np.arange(self.n), 'ok', ms=10)
        axs[0].plot(np.ones(self.n), np.arange(self.n), 'ok', ms=10)

        for S, c, offset in zip(
                (self.exc_synapses, self.inhib_synapses),
                ('b', 'r'),
                (0, self.n_e)):
            Ns = len(S.source)
            Nt = len(S.target)

            for i, j in zip(S.i, S.j):
                axs[0].plot([0, 1], [i + offset, j], '-', c=c)

            axs[0].set_xticks([0, 1], ['Source', 'Target'])
            axs[0].set_ylabel('Neuron index')
            axs[0].set_xlim(-0.1, 1.1)
            axs[0].set_ylim(-1, max(Ns, Nt))

            axs[1].plot(np.asarray(S.i) + offset, np.asarray(S.j), 'o', c=c,
                        alpha=0.5)
            axs[1].set_xlabel('Source index')
            axs[1].set_ylabel('Target index')

        # Overlay leader neuron
        for i, j in zip(self.exc_synapses.i, self.exc_synapses.j):
            if i != self.leader_neuron_idx:
                continue
            else:
                axs[0].plot([0, 1], [i, j], '-', c='yellow')

        fig.subplots_adjust(wspace=0.4)
        if show:
            fig.show()

        return fig, axs

    def plot_state(self, show=True):

        if self.spike_results is None:
            raise SimulationNotRunException('Error, no results yet. '
                                            'Did you run the simulation?')

        fig, axs = plt.subplots(2, 1, sharex=True)

        axs[0].imshow(self.state_results.v_mV,
                      aspect='auto',
                      interpolation='nearest',
                      cmap='Blues',
                      extent=(
                          self.state_results.t_ms[0],
                          self.state_results.t_ms[-1],
                          0,
                          self.state_results.v_mV.shape[0]
                      ))

        axs[0].set_xlabel('Time [ms]')
        axs[0].set_ylabel('Neuron')

        traces = self.state_results.v_mV.T

        axs[1].plot(self.state_results.t_ms, traces, c='blue')
        if self.leader_neuron_idx < traces.shape[1]:
            axs[1].plot(self.state_results.t_ms,
                        traces[:, self.leader_neuron_idx],
                        label='Leader neuron', c='yellow')

        axs[1].set_ylabel('Voltage [mV]')
        axs[1].legend()

        fig.subplots_adjust(hspace=0)

        if show:
            fig.show()

        return fig, axs

    def plot_spikes(self,
                    title=None,
                    t_range: tuple = None,
                    rate_range: tuple = None,
                    show=True,
                    rate_tick_step: float = 50, ):

        if self.spike_results is None or self.rate_results is None:
            raise SimulationNotRunException('Error, no results yet. '
                                            'Did you run the simulation?')

        fig = plt.figure()
        fig.suptitle(title)

        gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[4, 1])

        ax_spikes, ax_rates = gs.subplots(sharex="col")

        ax_spikes.plot(self.spike_results.as_array(), "|")
        ax_rates.plot(self.rate_results.as_array())

        ax_spikes.set_yticks([])

        ax_spikes.set_xlim(t_range)
        ax_rates.set_xlim(t_range)

        ax_rates.set_ylim(rate_range)
        ax_rates.set_xlabel("t [ms]")

        if rate_range:
            ax_rates.set_yticks(
                np.arange(
                    rate_range[0],
                    rate_range[1] + rate_tick_step,
                    rate_tick_step
                )
            )

        fig.subplots_adjust(hspace=0)

        if show:
            fig.show()

        return fig, (ax_spikes, ax_rates)
