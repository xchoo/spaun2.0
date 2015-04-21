from copy import deepcopy as copy
import numpy as np

import nengo
from nengo.networks import EnsembleArray
from nengo.dists import Choice
from nengo.dists import Uniform


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, mem_synapse=0.1,
                 fdbk_transform=1.0, input_transform=1.0, difference_gain=1.0,
                 gate_gain=3, reset_value=None, label=None, seed=None,
                 add_to_container=None, **mem_args):

        super(InputGatedMemory, self).__init__(label, seed, add_to_container)

        # Keep copy of network parameters
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.gate_gain = gate_gain
        self.mem_synapse = mem_synapse
        self.mem_args = copy(mem_args)
        self.input_transform = input_transform

        with self:
            self.input = nengo.Node(size_in=dimensions)

            # integrator to store value
            if np.isscalar(fdbk_transform):
                fdbk_matrix = np.eye(dimensions) * fdbk_transform
            else:
                fdbk_matrix = np.matrix(fdbk_transform)

            self.mem = EnsembleArray(n_neurons, dimensions, label="mem",
                                     **self.mem_args)
            nengo.Connection(self.mem.output, self.mem.input,
                             synapse=mem_synapse, transform=fdbk_matrix)
            self.output = self.mem.output

            # calculate difference between stored value and input
            self.diff = EnsembleArray(n_neurons, dimensions, label="diff",
                                      **self.mem_args)
            self.diff_input = self.diff.input

            nengo.Connection(self.input, self.diff.input, synapse=None,
                             transform=self.input_transform)
            nengo.Connection(self.mem.output, self.diff.input, transform=-1)

            # feed difference into integrator
            nengo.Connection(self.diff.output, self.mem.input,
                             transform=difference_gain, synapse=mem_synapse)

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                       intercepts=Uniform(0.5, 1))
            for e in self.diff.ensembles:
                nengo.Connection(self.gate, e.neurons,
                                 transform=[[-gate_gain]] * e.n_neurons)

        # No indenty! Not supposed to be within network context
        if reset_value is not None:
            make_resettable(self, reset_value)


class InputGatedCleanupMemory(nengo.Network):
    def __init__(self, n_neurons, dimensions, mem_synapse=0.1,
                 fdbk_transform=1.0, input_transform=1.0, difference_gain=1.0,
                 gate_gain=3, reset_value=None, cleanup_values=None,
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
        self.input_transform = np.dot(input_transform, cleanup_values)
        self.mem_synapse = mem_synapse
        self.mem_args = copy(mem_args)

        cu_args = copy(mem_args)
        cu_args['radius'] = 1
        cu_args['encoders'] = Choice([[1]])
        cu_args['intercepts'] = Uniform(0.3, 1)
        cu_args['eval_points'] = Uniform(0.4, 1.3)
        cu_args['n_eval_points'] = 5000

        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)

            self.mem = EnsembleArray(n_neurons, cleanup_values.shape[0],
                                     label="mem", **cu_args)
            # self.mem.add_output('thresh', function=lambda x: 1)
            # nengo.Connection(self.mem.thresh, self.mem.input,
            #                  synapse=mem_synapse)
            nengo.Connection(self.mem.output, self.mem.input,
                             synapse=mem_synapse)

            # calculate difference between stored value and input
            diff_args = copy(mem_args)
            diff_args['radius'] = 1
            self.diff = EnsembleArray(n_neurons, cleanup_values.shape[0],
                                      label="diff", **diff_args)
            self.diff_input = self.diff.input

            nengo.Connection(self.input, self.diff.input, synapse=None,
                             transform=self.input_transform)
            nengo.Connection(self.mem.output, self.diff.input, transform=-1)

            # feed difference into integrator
            nengo.Connection(self.diff.output, self.mem.input,
                             transform=difference_gain, synapse=mem_synapse)

            # connect cleanup to output
            # nengo.Connection(self.mem.thresh, self.output,
            #                  transform=cleanup_values.T, synapse=None)
            nengo.Connection(self.mem.output, self.output,
                             transform=cleanup_values.T, synapse=None)

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]),
                                       intercepts=Uniform(0.5, 1))
            for e in self.diff.ensembles:
                nengo.Connection(self.gate, e.neurons,
                                 transform=[[-gate_gain]] * e.n_neurons)

        # No indenty! Not supposed to be within network context
        if reset_value is not None:
            make_resettable(self, reset_value)


class InputGatedCleanupPlusMemory(nengo.Network):
    def __init__(self, n_neurons, dimensions, mem_synapse=0.1,
                 fdbk_transform=1.0, input_transform=1.0, difference_gain=1.0,
                 gate_gain=3, reset_value=None, cleanup_values=None,
                 label=None, seed=None, add_to_container=None, **mem_args):

        super(InputGatedCleanupPlusMemory, self).__init__(label, seed,
                                                          add_to_container)

        # Keep copy of network parameters
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.gate_gain = gate_gain
        self.input_transform = 1.0
        self.mem_synapse = mem_synapse
        self.mem_args = copy(mem_args)

        with self:
            # Set up input and output nodes
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)
            self.gate = nengo.Node(size_in=1)
            self.diff_input = nengo.Node(size_in=dimensions)

            self.mem = nengo.Network()
            with self.mem:
                self.mem_cleanup = \
                    InputGatedCleanupMemory(n_neurons, dimensions, mem_synapse,
                                            fdbk_transform, input_transform,
                                            difference_gain, gate_gain,
                                            reset_value, cleanup_values,
                                            **mem_args)
                self.mem_regular = \
                    InputGatedMemory(n_neurons, dimensions, mem_synapse,
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
                nengo.Connection(self.mem_cleanup.mem.output, e.neurons,
                                 transform=[[-gate_gain] *
                                            cleanup_values.shape[0]] *
                                 e.n_neurons)

        # No indenty! Not supposed to be within network context
        if reset_value is not None:
            make_resettable(self, reset_value)


def make_resettable(mem_network, reset_value):
    if np.isscalar(reset_value) and reset_value == 0:
        with mem_network as net:
            net.reset = nengo.Ensemble(net.n_neurons, 1, 
                                       encoders=Choice([[1]]),
                                       intercepts=Uniform(0.5, 1))
            for e in net.mem.all_ensembles:
                nengo.Connection(net.reset, e.neurons,
                                 transform=[[-net.gate_gain]] * e.n_neurons)
    else:
        if np.isscalar(reset_value):
            reset_value = np.matrix(np.ones(mem_network.dimensions) *
                                    reset_value)
        else:
            reset_value = np.matrix(reset_value)

        with mem_network as net:
            bias = nengo.Node(output=1)
            net.reset = nengo.Node(size_in=1)

            resetX = nengo.Ensemble(net.n_neurons, 1, encoders=Choice([[1]]),
                                    intercepts=Uniform(0.5, 1))
            resetN = nengo.Ensemble(net.n_neurons, 1, encoders=Choice([[1]]),
                                    intercepts=Uniform(0.5, 1))
            reset_gate = EnsembleArray(net.n_neurons, net.dimensions,
                                       label="reset gate", **net.mem_args)

            nengo.Connection(net.reset, resetX, synapse=None)
            nengo.Connection(net.reset, resetN, transform=-1, synapse=None)
            nengo.Connection(bias, resetN, synapse=None)

            nengo.Connection(bias, reset_gate.input, transform=reset_value.T,
                             synapse=None)
            nengo.Connection(net.input, reset_gate.input, transform=-1,
                             synapse=None)

            nengo.Connection(reset_gate.output, net.diff_input,
                             transform=net.input_transform)

            nengo.Connection(resetX, net.gate, transform=-1)
            for e in reset_gate.ensembles:
                nengo.Connection(resetN, e.neurons,
                                 transform=[[-net.gate_gain]] * e.n_neurons)
