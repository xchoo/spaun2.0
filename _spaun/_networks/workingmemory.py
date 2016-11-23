import numpy as np

import nengo
from nengo.networks import EnsembleArray
from nengo.dists import Choice
from nengo.dists import Exponential

from .assoc_mem import AssociativeMemory


def make_ensarray_func(n_neurons, dimensions, **ens_args):
    n_ensembles = ens_args.get('n_ensembles', dimensions)
    return EnsembleArray(n_neurons, n_ensembles, **ens_args)


def make_am_func(n_neurons, dimensions, input_vectors, **am_args):
    return AssociativeMemory(n_neurons=n_neurons, input_vectors=input_vectors,
                             **am_args)


def make_mem_network(net, n_neurons, dimensions, make_mem_func, make_mem_args,
                     make_diff_func, make_diff_args, mem_synapse=0.1,
                     fdbk_transform=1.0, input_transform=1.0,
                     difference_gain=1.0, gate_gain=3):
    with net:
        net.input = nengo.Node(size_in=dimensions)

        # integrator to store value
        if np.isscalar(fdbk_transform):
            fdbk_matrix = np.eye(dimensions) * fdbk_transform
        else:
            fdbk_matrix = np.matrix(fdbk_transform)

        net.mem = make_mem_func(n_neurons=n_neurons, dimensions=dimensions,
                                label="mem", **make_mem_args)
        if isinstance(net.mem, nengo.Network):
            mem_output = net.mem.output
            mem_input = net.mem.input
        else:
            mem_output = mem_input = net.mem

        nengo.Connection(mem_output, mem_input,
                         synapse=mem_synapse, transform=fdbk_matrix)

        # calculate difference between stored value and input
        net.diff = make_diff_func(n_neurons=n_neurons, dimensions=dimensions,
                                  label="Diff", **make_diff_args)

        if isinstance(net.diff, nengo.Network):
            net.diff_input = net.diff.input
            diff_output = net.diff.output
        else:
            net.diff_input = diff_output = net.diff

        nengo.Connection(net.input, net.diff_input, synapse=None,
                         transform=net.input_transform)
        nengo.Connection(mem_output, net.diff_input, transform=-1)

        # feed difference into integrator
        nengo.Connection(diff_output, mem_input,
                         transform=difference_gain, synapse=mem_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        # Note: A node is used for the input to make reset circuit more
        #       straightforward
        net.gate = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                  intercepts=Exponential(0.15, 0.5, 1),
                                  label='Gate')

        if isinstance(net.diff, nengo.Network):
            for e in net.diff.ensembles:
                nengo.Connection(net.gate, e.neurons,
                                 transform=[[-gate_gain]] * e.n_neurons)
        else:
            nengo.Connection(net.gate, net.diff.neurons,
                             transform=[[-gate_gain]] * e.n_neurons)

        # Make output
        net.output = net.mem.output


def make_resettable(net, n_neurons, dimensions, reset_value,
                    make_reset_func, make_reset_args, gate_gain=3):
    # Why have all this extra hardware to reset WM when inhibition will do
    # it?
    # - Makes the reset more reliable (storing a zero, rather than just
    #   wiping it out), so that reset signal can be shorter.
    if np.isscalar(reset_value):
        reset_value = np.matrix(np.ones(dimensions) * reset_value)
    else:
        reset_value = np.matrix(reset_value)

    with net:
        bias = nengo.Node(output=1, label='bias')
        net.reset = nengo.Node(size_in=1, label='reset')

        # Create resetX and resetN signals. resetX is to disable the WM gate
        # signal when reset is high. resetN is to disable the reset circuitry
        # when reset is low.
        resetX = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                intercepts=Exponential(0.15, 0.5, 1),
                                label='ResetX')
        resetN = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                intercepts=Exponential(0.15, 0.75, 1),
                                label='ResetN')
        resetN_delay = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                      intercepts=Exponential(0.15, 0.75, 1),
                                      label='ResetN delay')

        nengo.Connection(net.reset, resetX, synapse=None)
        # nengo.Connection(net.reset, resetN, transform=-1, synapse=None)
        nengo.Connection(resetX, resetN, transform=-gate_gain)
        # Note: gate_gain transform is to match net.gate -- so that the 'turn
        #       off' time is identical to net.gate
        nengo.Connection(bias, resetN, synapse=None)
        nengo.Connection(resetN, resetN_delay, synapse=0.01)

        # THe reset gate. Controls reset information going into the difference
        # population
        reset_gate = make_reset_func(n_neurons=n_neurons,
                                     dimensions=dimensions,
                                     label="Reset gate", **make_reset_args)

        if isinstance(reset_gate, nengo.Network):
            reset_gate_input = reset_gate.input
            reset_gate_output = reset_gate.output
        else:
            reset_gate_input = reset_gate_output = reset_gate

        # The desired reset value.
        nengo.Connection(bias, reset_gate_input, transform=reset_value.T,
                         synapse=None)
        # Need to negate whatever is being fed to the input of the WM (since
        # the WM difference population is going to be enabled).
        nengo.Connection(net.input, reset_gate_input,
                         transform=-net.input_transform, synapse=None)

        nengo.Connection(reset_gate_output, net.diff_input)
        # Note: Synapse is there to give slight delay for reset signal (to
        # gate) to dissipate.

        # Enable the WM difference population
        nengo.Connection(resetX, net.gate, transform=-gate_gain)

        # Disable the reset gate when reset signal is not active.
        if isinstance(reset_gate, nengo.Network):
            for e in reset_gate.ensembles:
                nengo.Connection(resetN_delay, e.neurons,
                                 transform=[[-gate_gain]] * e.n_neurons)
        else:
            nengo.Connection(resetN_delay, reset_gate.neurons,
                             transform=[[-gate_gain]] * e.n_neurons)


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, make_ens_func=make_ensarray_func,
                 mem_synapse=0.1, fdbk_transform=1.0, input_transform=1.0,
                 difference_gain=1.0, gate_gain=3, reset_value=None,
                 label=None, seed=None, add_to_container=None, **mem_args):

        super(InputGatedMemory, self).__init__(label, seed, add_to_container)

        # Keep copy of network parameters
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.gate_gain = gate_gain
        self.mem_synapse = mem_synapse
        self.input_transform = input_transform

        ens_args = dict(mem_args)

        make_mem_network(self, self.n_neurons, self.dimensions,
                         make_ens_func, ens_args, make_ens_func, ens_args,
                         mem_synapse, fdbk_transform, input_transform,
                         difference_gain, gate_gain)

        if reset_value is not None:
            make_resettable(self, self.n_neurons, self.dimensions,
                            reset_value, make_ens_func, ens_args,
                            gate_gain)


class InputGatedCleanupMemory(nengo.Network):
    def __init__(self, n_neurons, dimensions, make_ens_func=make_ensarray_func,
                 mem_synapse=0.1, fdbk_transform=1.0, input_transform=1.0,
                 difference_gain=1.0, gate_gain=3, reset_value=None,
                 cleanup_values=None, wta_output=False, wta_inhibit_scale=1,
                 label=None, seed=None, add_to_container=None, **mem_args):

        super(InputGatedCleanupMemory, self).__init__(label, seed,
                                                      add_to_container)

        if cleanup_values is None:
            raise ValueError('InputGatedCleanupMemory - cleanup_values must' +
                             ' be defined.')
        else:
            cleanup_values = np.matrix(cleanup_values)

        # Keep copy of network parameters
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.gate_gain = gate_gain
        self.mem_synapse = mem_synapse
        self.mem_args = dict(mem_args)
        self.input_transform = input_transform

        ens_args = dict(mem_args)

        make_mem_args = dict()
        make_mem_args['input_vectors'] = cleanup_values
        make_mem_args['threshold'] = ens_args.pop('threshold', 0.5)

        make_mem_network(self, self.n_neurons, self.dimensions,
                         make_am_func, make_mem_args, make_ens_func, ens_args,
                         mem_synapse, fdbk_transform, input_transform,
                         difference_gain, gate_gain)

        if wta_output:
            self.mem.add_wta_network(wta_inhibit_scale)

        if reset_value is not None:
            make_resettable(self, self.n_neurons, self.dimensions,
                            reset_value, make_ens_func, ens_args,
                            gate_gain)


class InputGatedCleanupPlusMemory(nengo.Network):
    def __init__(self, n_neurons, dimensions, make_ens_func=make_ensarray_func,
                 mem_synapse=0.1, fdbk_transform=1.0, input_transform=1.0,
                 difference_gain=1.0, gate_gain=3, reset_value=None,
                 cleanup_values=None, wta_output=False, wta_inhibit_scale=1,
                 label=None, seed=None, add_to_container=None, **mem_args):

        super(InputGatedCleanupPlusMemory, self).__init__(label, seed,
                                                          add_to_container)

        # Keep copy of network parameters
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.gate_gain = gate_gain
        self.input_transform = 1.0
        self.mem_synapse = mem_synapse
        self.mem_args = dict(mem_args)

        ens_args = dict(mem_args)

        with self:
            # Set up input and output nodes
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)
            self.gate = nengo.Node(size_in=1)
            self.diff_input = nengo.Node(size_in=dimensions)

            self.mem = nengo.Network()
            with self.mem:
                self.mem_cleanup = \
                    InputGatedCleanupMemory(n_neurons, dimensions,
                                            make_ensarray_func, mem_synapse,
                                            fdbk_transform, input_transform,
                                            difference_gain, gate_gain,
                                            reset_value, cleanup_values,
                                            wta_output, wta_inhibit_scale,
                                            **mem_args)
                self.mem_regular = \
                    InputGatedMemory(n_neurons, dimensions,
                                     make_ensarray_func, mem_synapse,
                                     fdbk_transform, input_transform,
                                     difference_gain, gate_gain,
                                     reset_value, **mem_args)

            nengo.Connection(self.input, self.mem_cleanup.input, synapse=None)
            nengo.Connection(self.input, self.mem_regular.input, synapse=None)
            nengo.Connection(self.diff_input, self.mem_cleanup.diff_input,
                             synapse=None,
                             transform=self.mem_cleanup.input_transform)
            nengo.Connection(self.diff_input, self.mem_regular.diff_input,
                             synapse=None,
                             transform=self.mem_regular.input_transform)
            nengo.Connection(self.mem_cleanup.output, self.output,
                             synapse=None)
            nengo.Connection(self.mem_regular.output, self.output,
                             synapse=None)
            nengo.Connection(self.gate, self.mem_cleanup.gate, synapse=None)
            nengo.Connection(self.gate, self.mem_regular.gate, synapse=None)

            for e in self.mem_regular.mem.ensembles:
                nengo.Connection(self.mem_cleanup.mem.output_utilities,
                                 e.neurons,
                                 transform=([[-gate_gain] *
                                             cleanup_values.shape[0]] *
                                            e.n_neurons))

        # No indentity! Not supposed to be within network context
        if reset_value is not None:
            make_resettable(self, self.n_neurons, self.dimensions,
                            reset_value, make_ensarray_func, ens_args,
                            gate_gain)
