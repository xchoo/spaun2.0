from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import vocab
from ..vocabs import pos_sp_strs
from ..vocabs import pos_mb_gate_sp_inds
from ..vocabs import pos_mb_rst_sp_inds


class InfoEncoding(Module):
    def __init__(self, label="Info Enc", seed=None, add_to_container=None):
        super(InfoEncoding, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        # Node that just outputs the INC vector
        self.inc_vec = nengo.Node(output=vocab['INC'].v)
        self.pos1_vec = nengo.Node(output=vocab['POS1'].v)

        # Memory block to store POS vector
        self.pos_mb = cfg.make_mem_block(label="POS MB", vocab=vocab,
                                         cleanup_keys=pos_sp_strs,
                                         reset_key='POS1')

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
        self.inputs = dict(default=(self.item_input, vocab))
        self.outputs = dict(default=(self.pos_output, vocab))

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

        # Set up connections from decoding module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.pos_mb_gate_bias, self.pos_mb.gate,
                             transform=4, synapse=0.01)
            nengo.Connection(parent_net.dec.pos_mb_gate_sig, self.pos_mb.gate,
                             transform=-4, synapse=0.01)
        else:
            warn("InfoEncoding Module - Cannot connect from 'dec'")
