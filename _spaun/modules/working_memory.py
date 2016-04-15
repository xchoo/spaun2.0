from warnings import warn
import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..utils import strs_to_inds
from ..vocabs import vocab, ps_state_sp_strs, ps_task_sp_strs, ps_dec_sp_strs
from ..vocabs import item_mb_gate_sp_inds, item_mb_rst_sp_inds
from ..vocabs import ave_mb_gate_sp_inds, ave_mb_rst_sp_inds

from .memory import WM_Generic_Network, WM_Averaging_Network


class WorkingMemory(Module):
    def __init__(self):
        super(WorkingMemory, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim, label='WM Module In Node')

        sp_add_matrix = (vocab['ADD'].get_convolution_matrix() *
                         (0.25 + 0.25 / cfg.mb_decaybuf_input_scale))
        # sp_add_matrix = (vocab['ADD'].get_convolution_matrix() * 0.5)

        self.num0_bias_node = nengo.Node(vocab.parse('POS1*ZER').v,
                                         label="POS1*ZER")

        self.gate_sig_bias = cfg.make_thresh_ens_net(label='Gate Sig Bias')
        # Bias the -1.5 neg_atn

        self.cnt_gate_sig = cfg.make_thresh_ens_net(0.5, label='Cnt Gate Sig')

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1_net = WM_Generic_Network(net_label="MB1", vocab=vocab,
                                          sp_add_matrix=sp_add_matrix)
        nengo.Connection(self.mem_in, self.mb1_net.input, synapse=None)
        nengo.Connection(self.num0_bias_node, self.mb1_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb1_net.gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb1_net.gate,
                         transform=1.5)

        self.mb1 = self.mb1_net.output

        # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
        self.mb2_net = WM_Generic_Network(net_label="MB2", vocab=vocab,
                                          sp_add_matrix=sp_add_matrix)
        nengo.Connection(self.mem_in, self.mb2_net.input, synapse=None)
        nengo.Connection(self.num0_bias_node, self.mb2_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb2_net.gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb2_net.gate,
                         transform=1.5)

        self.mb2 = self.mb2_net.output

        # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
        self.mb3_net = WM_Generic_Network(net_label="MB3", vocab=vocab,
                                          sp_add_matrix=sp_add_matrix)
        nengo.Connection(self.mem_in, self.mb3_net.input, synapse=None)
        nengo.Connection(self.num0_bias_node, self.mb3_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb3_net.gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb3_net.gate,
                         transform=1.5)

        self.mb3 = self.mb3_net.output

        # Memory block Ave (MBAve)
        self.mbave_net = WM_Averaging_Network(vocab=vocab)
        self.mbave = self.mbave_net.output

        # Define network inputs and outputs
        self.input = self.mem_in

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            # ###### MB1 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb1_net.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb1_net.gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb1_net.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MB2 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb2_net.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb2_net.gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb2_net.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MB3 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb3_net.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb3_net.gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb3_net.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MBAve ########
            nengo.Connection(p_net.vis.am_utilities[ave_mb_gate_sp_inds],
                             self.mbave_net.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(ave_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mbave_net.gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[ave_mb_rst_sp_inds],
                             self.mbave_net.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(ave_mb_rst_sp_inds)])
        else:
            warn("WorkingMemory Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            ps_state_mb_utils = p_net.ps.ps_state_utilities
            ps_task_mb_utils = p_net.ps.ps_task_utilities
            ps_dec_mb_utils = p_net.ps.ps_dec_utilities

            # ###### MB1 ########
            mb1_no_gate_strs = ['QAP', 'QAK', 'TRANS1', 'TRANS2', 'CNT0']
            mb1_no_gate_inds = strs_to_inds(mb1_no_gate_strs,
                                            ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb1_no_gate_inds],
                             self.mb1_net.gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_gate_inds)])

            mb1_no_reset_strs = ['QAP', 'QAK', 'TRANS1', 'CNT0', 'CNT1']
            mb1_no_reset_inds = strs_to_inds(mb1_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb1_no_reset_inds],
                             self.mb1_net.reset,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_reset_inds)])

            mb1_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb1_no_gate_inds = strs_to_inds(mb1_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb1_no_gate_inds],
                             self.mb1_net.gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_gate_inds)])

            mb1_sel_1_strs = ['CNT1']  # Use *ONE connection for CNT1 state
            mb1_sel_1_inds = strs_to_inds(mb1_sel_1_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb1_net.sel1)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb1_net.fdbk_gate)

            # ###### MB2 ########
            mb2_no_gate_strs = ['TRANS0', 'TRANS2', 'CNT1']
            mb2_no_gate_inds = strs_to_inds(mb2_no_gate_strs,
                                            ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb2_no_gate_inds],
                             self.mb2_net.gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_gate_inds)])

            mb2_no_reset_strs = ['QAP', 'QAK', 'TRANS1', 'TRANS2', 'CNT1']
            mb2_no_reset_inds = strs_to_inds(mb2_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb2_no_reset_inds],
                             self.mb2_net.reset,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_reset_inds)])

            mb2_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb2_no_gate_inds = strs_to_inds(mb2_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb2_no_gate_inds],
                             self.mb2_net.gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_gate_inds)])

            # ###### MB3 ########
            mb3_no_gate_strs = ['QAP', 'QAK', 'TRANS0', 'TRANS1']
            mb3_no_gate_inds = strs_to_inds(mb3_no_gate_strs,
                                            ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_no_gate_inds],
                             self.mb3_net.gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_gate_inds)])

            mb3_no_reset_strs = ['CNT1']
            mb3_no_reset_inds = strs_to_inds(mb3_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_no_reset_inds],
                             self.mb3_net.reset,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_reset_inds)])

            mb3_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb3_no_gate_inds = strs_to_inds(mb3_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb3_no_gate_inds],
                             self.mb3_net.gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_gate_inds)])

            mb3_sel_1_strs = ['CNT1']  # Use *ONE connection for CNT1 state
            mb3_sel_1_inds = strs_to_inds(mb3_sel_1_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_sel_1_inds],
                             self.mb3_net.sel1)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb3_net.fdbk_gate)

            mb3_sel_2_strs = ['CNT0']  # Use POS1*ONE connection for CNT0 state
            mb3_sel_2_inds = strs_to_inds(mb3_sel_2_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_sel_2_inds],
                             self.mb3_net.sel2)

            # ###### MBAVe ########
            mbave_no_gate_strs = ['QAP', 'QAK', 'TRANS0']
            mbave_no_gate_inds = strs_to_inds(mbave_no_gate_strs,
                                              ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mbave_no_gate_inds],
                             self.mbave_net.gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mbave_no_gate_inds)])

            mbave_do_reset_strs = ['X']
            mbave_do_reset_inds = strs_to_inds(mbave_do_reset_strs,
                                               ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mbave_do_reset_inds],
                             self.mbave_net.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(mbave_do_reset_inds)])

            # ###### Gate Signal Bias ######
            gate_sig_bias_enable_strs = ['CNT']  # Only enable gate signal bias
                                                 # for dec=CNT  # noqa
            gate_sig_bias_enable_inds = strs_to_inds(gate_sig_bias_enable_strs,
                                                     ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[gate_sig_bias_enable_inds],
                             self.gate_sig_bias.input,
                             transform=[[1] * len(gate_sig_bias_enable_inds)],
                             synapse=0.01)
        else:
            warn("WorkingMemory Module - Cannot connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.enc_output, self.mem_in)
        else:
            warn("WorkingMemory Module - Cannot connect from 'enc'")

        # Set up connections from transformation system module
        if hasattr(p_net, 'trfm'):
            nengo.Connection(p_net.trfm.output, self.mbave_net.input)
        else:
            warn("WorkingMemory Module - Cannot connect from 'trfm'")

        # Set up connections from motor module (for counting task)
        if hasattr(parent_net, 'mtr'):
            nengo.Connection(parent_net.mtr.ramp_50_75,
                             self.cnt_gate_sig.input, transform=2,
                             synapse=0.01)
        else:
            warn("WorkingMemory Module - Could not connect from 'mtr'")


class WorkingMemoryDummy(WorkingMemory):
    def __init__(self):
        super(WorkingMemoryDummy, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim, label='WM Module In Node')

        self.gate_sig_bias = nengo.Node(size_in=1)

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1 = \
            nengo.Node(output=vocab.parse('POS1*FOR+POS2*THR+POS3*FOR').v)
        self.mb1_net.gate = nengo.Node(size_in=1, label='MB1 Gate Node')
        self.mb1_net.reset = nengo.Node(size_in=1, label='MB1 Reset Node')

        self.sel_mb1_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=cfg.sp_dim,
                                            n_neurons=cfg.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb1_in.input0, synapse=None)

        # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
        self.mb2 = nengo.Node(output=vocab.parse('POS1*THR').v)
        self.mb2_gate = nengo.Node(size_in=1, label='MB2 Gate Node')
        self.mb2_reset = nengo.Node(size_in=1, label='MB2 Reset Node')

        self.sel_mb2_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=cfg.sp_dim,
                                            n_neurons=cfg.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb2_in.input0, synapse=None)

        # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
        self.mb3 = nengo.Node(output=vocab.parse('POS1*ONE').v)
        self.mb3_gate = nengo.Node(size_in=1, label='MB3 Gate Node')
        self.mb3_reset = nengo.Node(size_in=1, label='MB3 Reset Node')

        self.sel_mb3_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=cfg.sp_dim,
                                            n_neurons=cfg.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb3_in.input0, synapse=None)

        # Memory block Ave (MBAve)
        self.mbave_in = nengo.Node(size_in=cfg.sp_dim)
        self.mbave = nengo.Node(output=vocab.parse('~POS1').v)
        self.mbave_gate = nengo.Node(size_in=1)
        self.mbave_reset = nengo.Node(size_in=1)

        # Define network inputs and outputs
        # ## TODO: Fix this! (update to include selector and what not)
        self.input = self.mem_in
