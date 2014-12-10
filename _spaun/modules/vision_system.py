from warnings import warn

import nengo
from nengo.spa.module import Module

from .._spa import AssociativeMemory as AM
from .._spa import MemoryBlock as MB

from ..config import cfg
from ..vocabs import vis_vocab
from ..vocabs import item_mb_gate_sp_inds
from ..vision.lif_vision import LIFVision as LIFVisionNet
from ..vision.lif_vision import am_vis_sps
from ..vision.lif_vision import scales_data
from ..vision.lif_vision import vis_sps_radius

# --- Visual associative memory configurations ---
am_threshold = 0.5 * scales_data


class VisionSystem(Module):
    def __init__(self):
        super(VisionSystem, self).__init__()
        self.vis_net = LIFVisionNet(self)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.am = AM(am_vis_sps, vis_vocab, wta_output=True,
                     threshold=am_threshold, threshold_output=True,
                     inhibitable=True, inhibit_scale=5)
        nengo.Connection(self.vis_net.output, self.am.input, synapse=0.005)
        nengo.Connection(self.vis_net.neg_attention, self.am.inhibit,
                         synapse=0.005)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        cfg.vis_dim = am_vis_sps.shape[1]
        self.vis_mb = MB(cfg.n_neurons_mb * 2, cfg.vis_dim, gate_mode=2,
                         radius=vis_sps_radius, **cfg.mb_config)
        nengo.Connection(self.am.thresholded_utilities[item_mb_gate_sp_inds],
                         self.vis_mb.gate,
                         transform=[[cfg.mb_gate_scale] *
                                    len(item_mb_gate_sp_inds)])
        nengo.Connection(self.vis_net.output, self.vis_mb.input, synapse=0.01)
        nengo.Connection(self.vis_net.neg_attention,
                         self.vis_mb.gate, transform=-1, synapse=0.01)

        # Define network input and outputs
        self.input = self.vis_net.input
        self.output = self.am.output
        self.mb_output = self.vis_mb.output
        self.am_utilities = self.am.thresholded_utilities
        self.neg_attention = self.vis_net.neg_attention

        # Define module inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))

    def setup_connections(self, parent_net):
        # Set up connections from stimulus module
        if hasattr(parent_net, 'stim'):
            nengo.Connection(parent_net.stim.output, self.input)
        else:
            warn("Vision Module - Cannot connect from 'stim'")
