from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.distributions import Uniform
from nengo.utils.distributions import Choice
from nengo.networks import EnsembleArray

from ..config import cfg
from ..vocabs import vocab
from ..vocabs import pos_vocab
from ..vocabs import item_vocab
from ..vocabs import pos_mb_gate_sp_inds
from ..vocabs import pos_mb_rst_sp_inds


class InfoEncoding(Module):
    def __init__(self):
        super(InfoEncoding, self).__init__()

        # Node that just outputs the INC vector
        self.inc_vec = nengo.Node(output=vocab['INC'].v)
        self.pos1_vec = nengo.Node(output=vocab['POS1'].v)

        # Memory block to store POS vector
        self.pos_mb = cfg.make_mem_block(label="POS MB",
                                         cleanup_vecs=pos_vocab.vectors,
                                         reset_vec=pos_vocab['POS1'].v)

        # POS x INC
        nengo.Connection(self.pos_mb.output, self.pos_mb.input,
                         transform=vocab['INC'].get_convolution_matrix())

        # POS x ITEM
        self.item_cconv = cfg.make_cir_conv()
        nengo.Connection(self.pos_mb.output, self.item_cconv.A)

        self.enc_output = nengo.Node(size_in=cfg.sp_dim)
        nengo.Connection(self.item_cconv.output, self.enc_output)

        # Define network inputs and outputs
        self.pos_output = self.pos_mb.output
        self.item_input = self.item_cconv.B
        self.enc_output = self.item_cconv.output

        # Define module inputs and outputs
        self.inputs = dict(default=(self.item_input, item_vocab))
        self.outputs = dict(default=(self.pos_output, pos_vocab))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            nengo.Connection(parent_net.vis.am_utilities[pos_mb_gate_sp_inds],
                             self.pos_mb.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.pos_mb.gate, transform=-1, synapse=0.01)

            nengo.Connection(parent_net.vis.am_utilities[pos_mb_rst_sp_inds],
                             self.pos_mb.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_rst_sp_inds)])

            nengo.Connection(parent_net.vis.output, self.item_input)
            nengo.Connection(parent_net.vis.output, self.enc_output)
        else:
            warn("InfoEncoding Module - Cannot connect from 'vis'")
