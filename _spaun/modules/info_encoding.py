from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab

from .encoding import Pos_Inc_Network


class InfoEncoding(Module):
    def __init__(self, label="Information Enc", seed=None,
                 add_to_container=None):
        super(InfoEncoding, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        self.bias_node = nengo.Node(1, label='Bias')

        # ------ Common gate signal -----
        self.pos_gate = cfg.make_thresh_ens_net(0.25)
        nengo.Connection(self.bias_node, self.pos_gate.input, transform=1.2)

        # ------ Position (auto) incrementer network ------
        self.pos_inc = Pos_Inc_Network(vocab.pos, vocab.pos_sp_strs[0],
                                       vocab.inc_sp, reversable=True,
                                       threshold_gate_in=True)
        nengo.Connection(self.pos_gate.output, self.pos_inc.gate)

        # POS x ITEM
        self.item_cconv = cfg.make_cir_conv()
        nengo.Connection(self.pos_inc.output, self.item_cconv.A)

        self.enc_output = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(self.item_cconv.output, self.enc_output)

        # Increase the accumulator radius to account for increased magnitude
        # of added position vectors
        acc_radius = cfg.enc_mb_acc_radius_scale * cfg.get_optimal_sp_radius()

        # Memory block to store accumulated POS vectors (POSi-1 + POSi)
        self.pos_mb_acc = cfg.make_mem_block(label="POS MB ACC",
                                             vocab=vocab.pos,
                                             reset_key=0,
                                             radius=acc_radius,
                                             n_neurons=50,
                                             cleanup_mode=1,
                                             threshold_gate_in=True)
        nengo.Connection(self.pos_gate.output, self.pos_mb_acc.gate)
        nengo.Connection(self.pos_inc.output, self.pos_mb_acc.input)
        nengo.Connection(self.pos_mb_acc.output, self.pos_mb_acc.input)

        # REV state pos_inc gate bias (generates a pulse when VIS=QM and
        # DEC=REV to decrease POS in pos_inc by one -- because pos_inc always
        # holds the POS of the next item to be encoded).
        self.pos_inc_rev_gate_bias = cfg.make_thresh_ens_net()
        nengo.Connection(self.bias_node, self.pos_inc_rev_gate_bias.input,
                         transform=-1)
        nengo.Connection(self.pos_inc_rev_gate_bias.output, self.pos_inc.gate,
                         transform=cfg.mb_gate_scale)
        nengo.Connection(self.pos_inc_rev_gate_bias.output,
                         self.pos_inc.reverse, transform=cfg.mb_gate_scale)

        # Define network inputs and outputs
        self.pos_output = self.pos_inc.output
        self.pos_acc_output = self.pos_mb_acc.output
        self.item_input = self.item_cconv.B
        self.enc_output = self.item_cconv.output

        # Define module inputs and outputs
        self.inputs = dict(default=(self.item_input, vocab.item))
        self.outputs = dict(default=(self.pos_output, vocab.pos))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            # VIS ITEM Input
            nengo.Connection(parent_net.vis.output, self.item_input)

            # POS GATE control signal
            # pos_mb_gate_sp_vecs = \
            #     vocab.main.parse('+'.join(vocab.num_sp_strs)).v
            pos_mb_no_gate_sp_vecs = \
                vocab.main.parse('+'.join(vocab.ps_task_vis_sp_strs +
                                          vocab.misc_vis_sp_strs)).v
            nengo.Connection(parent_net.vis.output, self.pos_gate.input,
                             transform=[-cfg.mb_gate_scale *
                                        pos_mb_no_gate_sp_vecs])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.pos_gate.input,
                             transform=-cfg.mb_neg_attn_scale * 2.0,
                             synapse=0.02)

            # POS MB Control signals
            pos_mb_rst_sp_vecs = vocab.main.parse('A+OPEN+QM').v

            # nengo.Connection(parent_net.vis.output, self.pos_inc.gate,
            #                  transform=[cfg.mb_gate_scale *
            #                             pos_mb_gate_sp_vecs])
            # nengo.Connection(parent_net.vis.neg_attention,
            #                  self.pos_inc.gate, transform=-1.25,
            #                  synapse=0.01)

            nengo.Connection(parent_net.vis.output, self.pos_inc.reset,
                             transform=[pos_mb_rst_sp_vecs])

            # POS REV gate bias control signal
            pos_mb_rev_gate_sp_vecs = vocab.main.parse('QM').v
            nengo.Connection(parent_net.vis.output,
                             self.pos_inc_rev_gate_bias.input,
                             transform=[pos_mb_rev_gate_sp_vecs])

            # POS MB ACC Control signals
            pos_mb_acc_rst_sp_vecs = vocab.main.parse('A+OPEN').v

            # nengo.Connection(parent_net.vis.output, self.pos_mb_acc.gate,
            #                  transform=[cfg.mb_gate_scale *
            #                             pos_mb_gate_sp_vecs])
            # nengo.Connection(parent_net.vis.neg_attention,
            #                  self.pos_mb_acc.gate, transform=-1.25,
            #                  synapse=0.01)
            nengo.Connection(parent_net.vis.output, self.pos_mb_acc.reset,
                             transform=[pos_mb_acc_rst_sp_vecs])
        else:
            warn("InfoEncoding Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(parent_net, 'ps'):
            # Suppress the pos acc gate signal when in the decoding task stage
            pos_mb_acc_no_gate_sp_vecs = vocab.main.parse('DEC').v
            nengo.Connection(parent_net.ps.task, self.pos_mb_acc.gate,
                             transform=[-1.25 * pos_mb_acc_no_gate_sp_vecs])

            # Suppress the pos inc reset if dec == REV
            pos_mb_no_rst_sp_vecs = vocab.main.parse('REV').v
            nengo.Connection(parent_net.ps.dec, self.pos_inc.reset,
                             transform=[-1.25 * pos_mb_no_rst_sp_vecs])

            # Provide rev signal for pos_inc network
            pos_inc_rev_sp_vecs = vocab.main.parse('REV').v
            nengo.Connection(parent_net.ps.dec, self.pos_inc.reverse,
                             transform=[pos_inc_rev_sp_vecs])
            # But only when decoding
            pos_inc_no_rev_sp_vecs = vocab.main.parse('DEC').v
            nengo.Connection(self.bias_node, self.pos_inc.reverse,
                             transform=-1.25)
            nengo.Connection(parent_net.ps.task, self.pos_inc.reverse,
                             transform=[1.25 * pos_inc_no_rev_sp_vecs])

            # Provide rev signal for pos_inc_rev_gate_bias signal
            nengo.Connection(parent_net.ps.dec,
                             self.pos_inc_rev_gate_bias.input,
                             transform=[pos_inc_rev_sp_vecs])
        else:
            warn("InfoEncoding Module - Cannot connect from 'ps'")

        # Set up connections from decoding module
        if hasattr(parent_net, 'dec'):
            # nengo.Connection(parent_net.dec.pos_mb_gate_bias.output,
            #                  self.pos_inc.gate, transform=4, synapse=0.01)
            # nengo.Connection(parent_net.dec.pos_mb_gate_sig.output,
            #                  self.pos_inc.gate, transform=-4, synapse=0.01)
            nengo.Connection(parent_net.dec.pos_mb_gate_bias.output,
                             self.pos_inc.gate, transform=2.5, synapse=0.01)
            nengo.Connection(parent_net.dec.pos_mb_gate_sig.output,
                             self.pos_inc.gate, transform=-2.5, synapse=0.01)
        else:
            warn("InfoEncoding Module - Cannot connect from 'dec'")
