import nengo
from nengo.spa.module import Module

from ..config import cfg
from .._spa import AssociativeMemory as AM
from .._spa import MemoryBlock as MB
from .._vision.lif_vision import LIFVision as LIFVisionNet
from .._vision.lif_vision import am_vis_sps
from .._vision.lif_vision import scales_data
from .._vocab.vocabs import vis_vocab
from .._vocab.vocabs import item_mb_gate_sp_inds

# --- Visual associative memory configurations ---
am_threshold = 0.5 * scales_data


class LIFVision(Module):
    def __init__(self):
        super(LIFVision, self).__init__()
        self.vis_net = LIFVisionNet(self)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.am = AM(am_vis_sps, vis_vocab,
                     output_utilities=True, output_thresholded_utilities=True,
                     wta_output=True, threshold=am_threshold,
                     inhibitable=True, inhibit_scale=5)
        nengo.Connection(self.vis_net.output, self.am.input, synapse=0.005)
        nengo.Connection(self.vis_net.neg_attention, self.am.inhibit,
                         synapse=0.005)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        cfg.vis_dim = am_vis_sps.shape[1]
        self.vis_mb = MB(cfg.n_neurons_mb, cfg.vis_dim, gate_mode=2)
        nengo.Connection(self.am.thresholded_utilities[item_mb_gate_sp_inds],
                         self.vis_mb.gate,
                         transform=[[1] * len(item_mb_gate_sp_inds)])
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

    def connect_from_stimulus(self, stimulus):
        nengo.Connection(stimulus, self.input)
