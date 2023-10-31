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


@dataclass
class Monitors:
    state: StateMonitor
    spike: SpikeMonitor
    rate: PopulationRateMonitor

    def as_tuple(self):
        return self.state, self.spike, self.rate


class RecurrentNeuralNetwork:
    """Recurrent Neural Network of LIF neurons.

    Recurrent Neural Network (RNN) of Leaky-Integrate-and-Fire (LIF) neurons.
    Simulation based on Brunel, N. 2000 example of the ``brian2`` Python
    package: https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_2000.html#example-brunel-2000

    Experimental setup inspired by "Avalanche and edge-of-chaos criticality do
    not necessarily co-occur in neural networks" by Kanders et. al.:
    https://doi.org/10.1063/1.4978998

    The recurrent neural network has excitatory in inhibitory neurons. There
    are no self-connections.

    Attributes:
        tau: LIF neuron time constant.
        theta: LIF neuron threshold.
        V_r: Reset value for LIF neuron.
        tau_rp: Characteristic length of the refractory period in the LIF
          neurons.
        J: Base synaptic strength for LIF neuron.
        D: Synaptic delay for LIF neuron.
        n: Total number of neurons in the network.
        w: Weight scaling for synaptic strength.
        n_e: Number of excitatory neurons.
        p_c: Connection probability from one neuron to another.
        g: Relative strength of inhibitory synapses with respect to exitatory
          synapses
        gamma: Fraction of neurons in the network that are excitatory.
        p_ext: Probability for random activity in a neuron per time step.
        timestep_ms: Timestep in milliseconds. Defaults to 0.1 as per Brunel
          et. al.
        leader_neuron_idx: Index of the leader neuron. Note that the leader
          neuron is always excitatory.
        leader_rate: Frequency of input to the leader neuron in Hz.
        leader_weight: Magnitude of input to the leader neuron.
    """

    def __init__(self,
                 n: int = 100,
                 w: float = 10,
                 g: float = 3,
                 p_c: float = 0.04,
                 gamma=0.8,
                 p_ext: float = 6E-4,
                 seed: int = None,
                 leader_rate: float = 100,
                 timestep_ms: float = 0.1):

        """Initializes LIF RNN.

        Args:
            n: Number of neurons in the network. Defaults to 1000.
            w: Weight scaling for synaptic strength. Defaults to 10, which lies
              somewhat around the critical point.
            g: Relative strength of inhibitory synapses with respect to exitatory
              synapses. Defaults to 3 as seen in Kanders et. al.
            p_c: Connection probability. Defaults to 4% to make a sparse network
              as per Kanders et. al.
            gamma: Fraction of neurons in the network that are excitatory. Defaults
              to 0.8 as seen in Kanders et. al.
            p_ext: Probability for random activity in a neuron per time step.
              Adds noise. Defaults to 6E-4 as seen in Kanders et. al.
            seed: Seed with which to initialize random number generator. Used for
              network generation. Defaults to None (random).
            leader_rate: Frequency of input to the leader neuron in Hz. Defaults
              to 10 Hz.
            timestep_ms: Timestep in milliseconds. Defaults to 0.1 as per Brunel et.
              al.
        """

        # Parameters
        self.tau = 20 * ms
        self.theta = 20 * mV
        self.V_r = 10 * mV
        self.tau_rp = 2 * ms

        self.J = 0.1 * mV
        self.D = 1.5 * ms

        self.n = n
        self.w = w
        self.n_e = None
        self.p_c = p_c
        self.g = 3
        self.gamma = gamma
        self.p_ext = p_ext
        self.timestep_ms = timestep_ms
        self.leader_neuron_idx = None
        self.leader_rate = leader_rate

        # Brain2 Objects
        self._neurons = None
        self._synapses = None

        self._monitors = None

        self._net: Optional[Network] = None

        self.state_results: Optional[StateResults] = None
        self.spike_results: Optional[SpikeResults] = None
        self.rate_results: Optional[RateResults] = None

        self.seed = seed
        self._setup_network(seed)

    def sim(self, sim_time_ms):

        tau = self.tau
        tau_rp = self.tau_rp
        V_r = self.V_r
        theta = self.theta

        w = self.w
        g = self.g
        D = self.D
        J = self.J

        self._net.run(sim_time_ms * ms, report='text')

        self.rate_results = RateResults(self._monitors.rate)
        self.spike_results = SpikeResults(self._monitors.spike)
        self.state_results = StateResults(self._monitors.state)

        return 1

    def store(self, *args, **kwargs):
        self._net.store(*args, **kwargs)
        return self

    def restore(self, *args, **kwargs):
        self._net.restore(*args, **kwargs)
        return self

    def _setup_network(self, seed: int = None):

        # Set random number generator seed (optional)
        np.random.seed(seed)

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

        self._synapses = self._get_synapses(neurons, self.n_e)
        nu_ext = 1 * self.theta / (self.J * C_E * self.tau)

        external_poisson_input = PoissonInput(
            target=neurons, target_var="v", N=C_E, rate=nu_ext*0.8,
            weight=self.J
        )

        idx = np.random.randint(0, self.n_e)
        self.leader_neuron_idx = idx
        leading_neuron = neurons[idx:idx + 1]

        leading_neuron_input = PoissonInput(
            target=leading_neuron,
            target_var="v", N=1, rate=self.leader_rate * Hz,
            weight=self.theta
        )

        self._monitors = self.setup_monitors(neurons)
        self._neurons = neurons

        self._net = Network(collect())
        self._net.add(self._monitors.as_tuple())
        self._net.add(self._synapses)

        # Clear seeding
        np.random.seed(None)

    def _get_synapses(self, neurons, N_E):

        excitatory_neurons = neurons[:N_E]
        inhibitory_neurons = neurons[N_E:]

        exc_synapses = Synapses(excitatory_neurons,
                                target=neurons,
                                on_pre="v += J*w",
                                delay=self.D,
                                name="exc")
        exc_synapses.connect(p=self.p_c, condition='i!=j')
        inhib_synapses = Synapses(inhibitory_neurons,
                                  target=neurons,
                                  on_pre="v += -g*J*w",
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
        C_E = self.p_c * N_E

        return C_E, N_E

    def get_spike_distribution(self, delta_t_ms: float = 1):
        t_ms = self.spike_results.t_ms

        num_bins = int(t_ms[-1] / delta_t_ms)

        spikes, bins = np.histogram(t_ms, num_bins)
        spike_range = np.arange(0, np.max(spikes), 1)
        # plt.plot(spikes)
        # plt.show()

        counts, _ = np.histogram(spikes, bins=spike_range)

        mask = counts != 0

        return spike_range[:-1][mask], counts[mask]

    def plot_spike_distribution(self, delta_t_ms=0.5, show=True):
        fig, ax = plt.subplots(1, 1)

        spikes, counts = self.get_spike_distribution(delta_t_ms=delta_t_ms)

        log_spikes = np.log10(spikes)
        log_counts = np.log10(counts)

        a, b = np.polyfit(log_spikes[1:], log_counts[1:], 1)

        x = np.linspace(spikes[0], spikes[1])
        y = np.power(10, a * x + b)

        ax.loglog(spikes, counts, '.-', c='b')
        ax.loglog(np.power(10, x), y, '--', c='r')

        ax.text(np.power(10, np.median(x)), np.median(y)*1.1, r'$\alpha\approx$' + f'{-a:.2f}', c='r')
        ax.set_xlabel('S [spikes]')
        ax.set_ylabel('counts')

        if show:
            fig.show()

        return fig, ax

    def set_w(self, w):
        self.w = w

    def get_w(self):
        return self.w

    def setup_monitors(self, neurons, i=-1) -> Monitors:
        rate_monitor = PopulationRateMonitor(neurons)
        # record from the first i excitatory neurons
        spike_monitor = SpikeMonitor(neurons[:i])
        state_monitor = StateMonitor(neurons[:i], 'v', record=True)

        return Monitors(state=state_monitor, spike=spike_monitor,
                        rate=rate_monitor)

    def plot_connectivity(self, show=True):
        if self._synapses is None:
            raise SimulationNotRunException('Error, no results yet. '
                                            'Did you run the simulation?')

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(np.zeros(self.n), np.arange(self.n), 'ok', ms=10)
        axs[0].plot(np.ones(self.n), np.arange(self.n), 'ok', ms=10)

        for S, c, offset in zip(
                self._synapses,
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
        for i, j in zip(self._synapses[0].i, self._synapses[0].j):
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
            fig.clf()

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

        ax_spikes.plot(self.spike_results.t_ms, self.spike_results.i, "|")
        ax_rates.plot(self.rate_results.t_ms, self.rate_results.rate)

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
