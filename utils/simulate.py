from brian2 import *
import matplotlib.pyplot as plt
import numpy as np


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
        rate_monitor: Stores instantaneous spiking rates of groups of neurons.
        spike_monitor: Stores spikes from individual neurons.
    """

    neuron_params = {
        'tau_ms': 20,
        'theta_mV': 20,
        'V_r_mV': 10,
        'tau_rp_ms': 2,
    }

    synapse_params = {
        'J_mV': 0.1,
        'D_ms': 1.5,
    }

    sim_params = {
        'dt_ms': 0.1
    }

    def __init__(self, n: int = 128, g: float = 3, p_c: float = 0.04, gamma=0.8,
                 p_ext: float = 6E-4, nu_ext_over_nu_thr = 0.9):

        self.n = n
        self.p_c = p_c
        self.g = 3
        self.gamma = gamma
        self.p_ext = p_ext

        self.nu_ext_over_nu_thr = nu_ext_over_nu_thr

        self.v = None
        self.t = None
        self.state_data = None

    def sim(self, sim_time_ms):

        # neuron parameters
        tau = self.neuron_params['tau_ms'] * ms
        theta = self.neuron_params['theta_mV'] * mV
        V_r = self.neuron_params['V_r_mV'] * mV
        tau_rp = self.neuron_params['tau_rp_ms'] * ms

        # synapse parameters
        g = self.g
        J = self.synapse_params['J_mV'] * mV
        D = self.synapse_params['D_ms'] * ms

        sim_time = sim_time_ms * ms

        # network parameters
        N_E = round(self.gamma * self.n)
        N_I = self.n - N_E

        epsilon = 0.1
        C_E = epsilon * N_E
        C_ext = C_E

        # external stimulus
        nu_thr = theta / (J * C_E * tau)

        # Noise rate
        rate = self.p_ext / (self.sim_params['dt_ms'] * 1E-3) * Hz

        defaultclock.dt = 0.1 * ms

        neurons = NeuronGroup(self.n,
                              """
                              dv/dt = -v/tau : volt (unless refractory)
                              """,
                              threshold="v > theta",
                              reset="v = V_r",
                              refractory=tau_rp,
                              method="exact",
                              )

        excitatory_neurons = neurons[:N_E]
        inhibitory_neurons = neurons[N_E:]

        exc_synapses = Synapses(excitatory_neurons,
                                target=neurons,
                                on_pre="v += J",
                                delay=D)
        exc_synapses.connect(p=self.gamma, condition='i!=j')

        inhib_synapses = Synapses(inhibitory_neurons,
                                  target=neurons,
                                  on_pre="v += -g*J",
                                  delay=D)
        inhib_synapses.connect(p=self.gamma,
                               condition='i!=j')  # No self connections

        nu_ext = self.nu_ext_over_nu_thr * nu_thr

        external_poisson_input = PoissonInput(
            target=neurons, target_var="v", N=self.n, rate=rate,
            weight=J
        )

        leading_idx = np.random.randint(0, N_E)
        leading_neuron = excitatory_neurons[leading_idx:leading_idx+1]

        leading_neuron_input = PoissonInput(
            target=leading_neuron,
            target_var="v", N=1, rate=5 * Hz,
            weight=10 * mV
        )

        rate_monitor = PopulationRateMonitor(neurons)

        # record from the first 50 excitatory neurons
        spike_monitor = SpikeMonitor(excitatory_neurons[:50])

        state_monitor = StateMonitor(neurons, 'v', record=True)

        net = Network(collect())

        net.run(sim_time, report='text')

        self.v = state_monitor.v / mV
        self.t = state_monitor.t / ms


    def plot(self,
        fig,
        title=None,
        t_range: tuple = None,
        rate_range: tuple = None,
        rate_tick_step: float = 50,):

        fig.suptitle(title)

        gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[4, 1])

        ax_spikes, ax_rates = gs.subplots(sharex="col")

        ax_spikes.plot(self.spike_monitor.t / ms, self.spike_monitor.i, "|")
        ax_rates.plot(self.rate_monitor.t / ms, self.rate_monitor.rate / Hz)

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

        return fig