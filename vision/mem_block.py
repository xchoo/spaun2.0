import math

import nengo
from nengo.spa.module import Module
from nengo.networks import InputGatedMemory as WM


class MemoryBlock(Module):
    def __init__(self, n_neurons_per_ensemble=50, dimensions=1, radius=None,
                 gate_mode=1, reset_mode=3, **mem_args):
        if radius is None:
            radius = 3.5 / math.sqrt(dimensions)

        self.gate = nengo.Ensemble(n_neurons_per_ensemble, 1, label="gate")
        self.reset = nengo.Node(size_in=1)

        self.mem1 = WM(n_neurons_per_ensemble, dimensions, radius=radius,
                       difference_gain=5, **mem_args)
        self.mem2 = WM(n_neurons_per_ensemble, dimensions, radius=radius,
                       difference_gain=6, gate_gain=3, **mem_args)

        # gate_modes:
        # - 1: Gate mem1 on gate high, gate mem2 on gate low (default)
        # - 2: Gate mem1 on gate low, gate mem2 on gate high
        if gate_mode == 1:
            nengo.Connection(self.gate, self.mem1.gate, synapse=0.005)
            nengo.Connection(self.gate, self.mem2.gate, synapse=0.005,
                             function=lambda x: 1 - x)
        else:
            nengo.Connection(self.gate, self.mem1.gate, synapse=0.005,
                             function=lambda x: 1 - x)
            nengo.Connection(self.gate, self.mem2.gate, synapse=0.005)

        # reset_modes:
        # - 1: Reset only mem1
        # - 2: Reset only mem2
        # - 3: Reset both mem1 and mem2
        if reset_mode & 1:
            nengo.Connection(self.reset, self.mem1.reset_node, synapse=None)
        if reset_mode & 2:
            nengo.Connection(self.reset, self.mem2.reset_node, synapse=None)

        nengo.Connection(self.mem1.output, self.mem2.input, synapse=0.005)

        self.input = self.mem1.input
        self.output = self.mem2.output
