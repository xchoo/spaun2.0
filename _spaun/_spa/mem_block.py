import math

import nengo
from nengo.spa.module import Module
from nengo.utils.distributions import Choice
from nengo.utils.distributions import Uniform

# from .._networks import InputGatedMemory as WM
from .._networks.workingmemory_2_0 import InputGatedMemory as WM


class MemoryBlock(Module):
    def __init__(self, n_neurons, dimensions,
                 radius=None, gate_mode=1, reset_mode=3, **mem_args):
        if radius is None:
            radius = 3.5 / math.sqrt(dimensions)

        if n_neurons == nengo.Default:
            n_neurons = nengo.config[nengo.Ensemble].n_neurons

        # Note: Both gate & gateN are needed here to produce dead-zero
        #       (no neural activity) when WM is non-gated. So a preceeding
        #       ensemble needs to generate the full range values to be fed into
        #       these two.
        self.gate = nengo.Ensemble(n_neurons, 1, label="gate")
        self.gateX = nengo.Ensemble(n_neurons, 1, label="gateX",
                                    intercepts=Uniform(0.4, 1),
                                    encoders=Choice([[1]]))
        self.gateN = nengo.Ensemble(n_neurons, 1, label="gateN",
                                    intercepts=Uniform(0.4, 1),
                                    encoders=Choice([[1]]))
        nengo.Connection(self.gate, self.gateX, function=lambda x: x)
        nengo.Connection(self.gate, self.gateN, function=lambda x: 1 - x)

        self.reset = nengo.Node(size_in=1)

        self.mem1 = WM(n_neurons, dimensions, radius=radius,
                       gate_gain=3, **mem_args)
        self.mem2 = WM(n_neurons, dimensions, radius=radius,
                       gate_gain=3, **mem_args)

        # gate_modes:
        # - 1: Gate mem1 on gate high, gate mem2 on gate low (default)
        # - 2: Gate mem1 on gate low, gate mem2 on gate high
        if gate_mode == 1:
            nengo.Connection(self.gateX, self.mem1.gate)
            nengo.Connection(self.gateN, self.mem2.gate)
        else:
            nengo.Connection(self.gateN, self.mem1.gate)
            nengo.Connection(self.gateX, self.mem2.gate)

        # reset_modes:
        # - 1: Reset only mem1
        # - 2: Reset only mem2
        # - 3: Reset both mem1 and mem2
        if reset_mode & 1:
            nengo.Connection(self.reset, self.mem1.reset, synapse=None)
        if reset_mode & 2:
            nengo.Connection(self.reset, self.mem2.reset, synapse=None)

        nengo.Connection(self.mem1.output, self.mem2.input, synapse=0.005)

        self.input = self.mem1.input
        self.output = self.mem2.output
