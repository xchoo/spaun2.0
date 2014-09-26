import numpy as np

import nengo
from nengo.networks import EnsembleArray


class InputGatedMemory(nengo.Network):
    """Stores a given vector in memory, with input controlled by a gate."""

    def __init__(self, n_neurons, dimensions, mem_synapse=0.1, fdbk_scale=1.0,
                 difference_gain=1.0, gate_gain=10, reset_gain=3,
                 **mem_args):

        self.input = nengo.Node(size_in=dimensions)
        self.output = nengo.Node(size_in=dimensions)

        # integrator to store value
        self.mem = EnsembleArray(n_neurons, dimensions, label="mem",
                                 **mem_args)
        nengo.Connection(self.mem.output, self.mem.input, synapse=mem_synapse,
                         transform=np.eye(dimensions) * fdbk_scale)

        # calculate difference between stored value and input
        self.diff = EnsembleArray(n_neurons, dimensions, label="diff")
        nengo.Connection(self.input, self.diff.input, synapse=None)
        nengo.Connection(self.mem.output, self.diff.input,
                         transform=np.eye(dimensions) * -1)

        # feed difference into integrator
        nengo.Connection(self.diff.output, self.mem.input,
                         transform=np.eye(dimensions) * difference_gain,
                         synapse=mem_synapse)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        self.gate = nengo.Node(size_in=1)
        for e in self.diff.ensembles:
            nengo.Connection(self.gate, e.neurons,
                             transform=[[-gate_gain]] * e.n_neurons)

        # reset input (if reset=1, remove all values stored, and set values
        # to 0)
        self.reset_node = nengo.Node(size_in=1)
        for e in self.mem.ensembles:
            nengo.Connection(self.reset_node, e.neurons,
                             transform=[[-reset_gain]] * e.n_neurons)

        nengo.Connection(self.mem.output, self.output, synapse=None)

