from copy import deepcopy as copy
import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.dists import Choice
from nengo.dists import Uniform

from .._networks import InputGatedMemory as WM
from .._networks import InputGatedCleanupMemory as WMC
from .._networks import InputGatedCleanupPlusMemory as WMCP


class MemoryBlock(Module):
    def __init__(self, n_neurons, dimensions, vocab,
                 radius=None, gate_mode=1, reset_mode=3, cleanup_mode=0,
                 cleanup_keys=None, reset_key=None, label=None, seed=None,
                 add_to_container=None, **mem_args):
        super(MemoryBlock, self).__init__(label, seed, add_to_container)

        if radius is None:
            radius = 3.5 / np.sqrt(dimensions)

        if n_neurons == nengo.Default:
            n_neurons = nengo.config[nengo.Ensemble].n_neurons

        if cleanup_keys is not None:
            cleanup_vecs = vocab.create_subset(cleanup_keys).vectors
        elif cleanup_mode != 0:
            cleanup_vecs = vocab.vectors
        else:
            cleanup_vecs = None

        if isinstance(reset_key, str):
            reset_vec = vocab.parse(reset_key).v
        elif reset_key is not None and reset_key != 0:
            reset_vec = np.ones(dimensions) * reset_key
        else:
            reset_vec = reset_key

        if reset_key is None:
            reset_mode = 0

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

            wm_args = copy(mem_args)
            wm_args['radius'] = radius
            wm_args['gate_gain'] = mem_args.get('gate_gain', 5)
            wm_args['difference_gain'] = mem_args.get('difference_gain', 5)

            # cleanup_mode:
            # - 0 (or cleanup_vecs == None): No cleanup
            # - 1: Cleanup memory vectors using provided vectors. Only store
            #      vectors that can be cleaned up.
            # - 2: Cleanup memory vectors using provided vectors but also allow
            #      vectors that do not match any of the cleanup vectors to be
            #      stored as well.
            if cleanup_vecs is None or cleanup_mode == 0:
                self.mem1 = WM(n_neurons, dimensions, radius=radius,
                               reset_value=reset_vec, **mem_args)
                self.mem2 = WM(n_neurons, dimensions, radius=radius,
                               reset_value=reset_vec, **mem_args)
            elif cleanup_mode == 1:
                self.mem1 = WMC(n_neurons, dimensions, radius=radius,
                                cleanup_values=cleanup_vecs,
                                reset_value=reset_vec, **mem_args)
                self.mem2 = WMC(n_neurons, dimensions, radius=radius,
                                cleanup_values=cleanup_vecs,
                                reset_value=reset_vec, **mem_args)
            else:
                self.mem1 = WMCP(n_neurons, dimensions, radius=radius,
                                 cleanup_values=cleanup_vecs,
                                 reset_value=reset_vec, **mem_args)
                self.mem2 = WMCP(n_neurons, dimensions, radius=radius,
                                 cleanup_values=cleanup_vecs,
                                 reset_value=reset_vec, **mem_args)

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

        # Configure SPA default input and output vocabularies
        self.inputs = dict(default=(self.input, vocab))
        self.outputs = dict(default=(self.output, vocab))
