from warnings import warn

import nengo
from nengo.spa import Vocabulary
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._spa import MemoryBlock as MB

from ..config import cfg
from ..vocabs import vocab, vis_vocab
from ..vocabs import item_mb_gate_sp_inds
from ..vision.lif_vision import LIFVision as LIFVisionNet
from ..vision.lif_vision import am_vis_sps
from ..vision.lif_vision import am_threshold
from ..vision.lif_vision import vis_sps_scale


class VisionSystem(Module):
    def __init__(self, label="Vision Sys", seed=None, add_to_container=None):
        super(VisionSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        self.vis_net = LIFVisionNet(self)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.am = cfg.make_assoc_mem(am_vis_sps, vis_vocab.vectors,
                                     threshold=am_threshold,
                                     inhibitable=True, inhibit_scale=5)
        nengo.Connection(self.vis_net.output, self.am.input, synapse=0.005)
        nengo.Connection(self.vis_net.neg_attention, self.am.inhibit,
                         synapse=0.005)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        cfg.vis_dim = am_vis_sps.shape[1]
        vis_sp_vocab = Vocabulary(cfg.vis_dim)
        self.vis_mb = MB(cfg.n_neurons_mb * 2, cfg.vis_dim, gate_mode=2,
                         vocab=vis_sp_vocab, **cfg.mb_config)
        nengo.Connection(self.am.thresholded_utilities[item_mb_gate_sp_inds],
                         self.vis_mb.gate,
                         transform=[[cfg.mb_gate_scale] *
                                    len(item_mb_gate_sp_inds)])
        nengo.Connection(self.vis_net.output, self.vis_mb.input,
                         transform=vis_sps_scale, synapse=0.01)
        nengo.Connection(self.vis_net.neg_attention,
                         self.vis_mb.gate, transform=-1, synapse=0.01)

        # Define network input and outputs
        self.input = self.vis_net.input
        self.output = self.am.output
        self.mb_output = self.vis_mb.output
        self.am_utilities = self.am.thresholded_utilities
        self.neg_attention = self.vis_net.neg_attention

        # Define module inputs and outputs
        self.outputs = dict(default=(self.output, vocab))

    def setup_connections(self, parent_net):
        # Set up connections from stimulus module
        if hasattr(parent_net, 'stim'):
            nengo.Connection(parent_net.stim.output, self.input)
        else:
            warn("Vision Module - Cannot connect from 'stim'")
