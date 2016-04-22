from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import vocab, pos_vocab, pos_sp_strs
from ..vocabs import pos_mb_gate_sp_inds
from ..vocabs import pos_mb_rst_sp_inds, pos_mb_acc_rst_sp_inds

from .encoding import Pos_Inc_Network


class InfoEncoding(Module):
    def __init__(self, label="Info Enc", seed=None, add_to_container=None):
        super(InfoEncoding, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        self.pos_inc = Pos_Inc_Network()

        # POS x ITEM
        self.item_cconv = cfg.make_cir_conv()
        nengo.Connection(self.pos_inc.output, self.item_cconv.A)

        self.enc_output = nengo.Node(size_in=cfg.sp_dim)
        nengo.Connection(self.item_cconv.output, self.enc_output)

        # Increase the accumulator radius to account for increased magnitude
        # of added position vectors
        acc_radius = cfg.enc_mb_acc_radius_scale * cfg.get_optimal_sp_radius()

        # Memory block to store accumulated POS vectors (POSi-1 + POSi)
        self.pos_mb_acc = cfg.make_mem_block(label="POS MB ACC",
                                             vocab=pos_vocab,
                                             reset_key=0,
                                             radius=acc_radius,
                                             n_neurons=100)  # ,
                                             # cleanup_keys=pos_sp_strs,
                                             # cleanup_mode=2)
        nengo.Connection(self.pos_inc.output, self.pos_mb_acc.input)
        nengo.Connection(self.pos_mb_acc.output, self.pos_mb_acc.input)

        # Define network inputs and outputs
        self.pos_output = self.pos_inc.output
        self.pos_acc_output = self.pos_mb_acc.output
        self.item_input = self.item_cconv.B
        self.enc_output = self.item_cconv.output

        # Define module inputs and outputs
        self.inputs = dict(default=(self.item_input, vocab))
        self.outputs = dict(default=(self.pos_output, vocab))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            vis_am_utils = parent_net.vis.am_utilities

            # POS MB Control signals
            nengo.Connection(vis_am_utils[pos_mb_gate_sp_inds],
                             self.pos_inc.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.pos_inc.gate, transform=-1.25,
                             synapse=0.01)

            nengo.Connection(vis_am_utils[pos_mb_rst_sp_inds],
                             self.pos_inc.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_rst_sp_inds)])

            # POS MB ACC Control signals
            nengo.Connection(vis_am_utils[pos_mb_gate_sp_inds],
                             self.pos_mb_acc.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.pos_mb_acc.gate, transform=-1.25,
                             synapse=0.01)

            nengo.Connection(vis_am_utils[pos_mb_acc_rst_sp_inds],
                             self.pos_mb_acc.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_acc_rst_sp_inds)])

            # VIS ITEM Input
            nengo.Connection(parent_net.vis.output, self.item_input)

            # TODO: Fix no resetting for REV recall
            # - Disable reset during QM
            # - Set INC selector to ~INC

            # Encode item in encoding (POSxITEM + ITEM)
            # nengo.Connection(parent_net.vis.output, self.enc_output,
            #                  synapse=None)
        else:
            warn("InfoEncoding Module - Cannot connect from 'vis'")

        # Set up connections from decoding module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.pos_mb_gate_bias.output,
                             self.pos_inc.gate, transform=5, synapse=0.05)
            nengo.Connection(parent_net.dec.pos_mb_gate_sig.output,
                             self.pos_inc.gate, transform=-4, synapse=0.01)
        else:
            warn("InfoEncoding Module - Cannot connect from 'dec'")

        # TODO: Add connection from PS to do REV recall
