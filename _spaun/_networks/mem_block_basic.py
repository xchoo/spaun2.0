import numpy as np

import nengo
from nengo.dists import Choice
from nengo.dists import Uniform

from nengo.networks import InputGatedMemory as WM


class MemoryBlock(nengo.Network):
    def __init__(self, n_neurons, dimensions, radius=None, gate_mode=1,
                 reset_mode=3, label=None, seed=None, add_to_container=None,
                 **mem_args):
        super(MemoryBlock, self).__init__(label, seed, add_to_container)

        if radius is None:
            radius = 3.5 / np.sqrt(dimensions)

        if n_neurons == nengo.Default:
            n_neurons = 100

        with self:
            # Note: Both gate & gateN are needed here to produce dead-zero
            #       (no neural activity) when WM is non-gated. So a preceeding
            #       ensemble needs to generate the full range values to be fed
            #       into these two.
            bias_node = nengo.Node(output=1)
            self.gate = nengo.Node(size_in=1, label="gate")

            self.gateX = nengo.Ensemble(n_neurons, 1, label="gateX",
                                        intercepts=Uniform(0.5, 1),
                                        encoders=Choice([[1]]))
            self.gateN = nengo.Ensemble(n_neurons, 1, label="gateN",
                                        intercepts=Uniform(0.5, 1),
                                        encoders=Choice([[1]]))
            nengo.Connection(self.gate, self.gateX)
            nengo.Connection(self.gate, self.gateN, transform=-1)
            nengo.Connection(bias_node, self.gateN)

            wm_args = dict(mem_args)
            wm_args['difference_gain'] = mem_args.get('difference_gain', 15)

            wm_config = nengo.Config(nengo.Ensemble)
            wm_config[nengo.Ensemble].radius = radius

            with wm_config:
                self.mem1 = WM(n_neurons, dimensions, **wm_args)
                self.mem2 = WM(n_neurons, dimensions, **wm_args)

            # gate_modes:
            # - 1: Gate mem1 on gate high, gate mem2 on gate low (default)
            # - 2: Gate mem1 on gate low, gate mem2 on gate high
            if gate_mode == 1:
                gateX = self.gateX
                gateN = self.gateN
            else:
                gateX = self.gateN
                gateN = self.gateX

            nengo.Connection(gateX, self.mem1.gate)
            nengo.Connection(gateN, self.mem2.gate)

            # reset_modes:
            # - 1: Reset only mem1
            # - 2: Reset only mem2
            # - 3: Reset both mem1 and mem2
            if reset_mode:
                self.reset = nengo.Node(size_in=1)
            if reset_mode & 1:
                nengo.Connection(self.reset, self.mem1.reset, synapse=None)
            if reset_mode & 2:
                nengo.Connection(self.reset, self.mem2.reset, synapse=None)

            nengo.Connection(self.mem1.output, self.mem2.input, synapse=0.005)

            # Input and output nodes
            self.input = self.mem1.input
            self.output = self.mem2.output
