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


class WorkingMemory(Module):
    def __init__(self):
        super(WorkingMemory, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        self.bias_node = nengo.Node(1)

        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim, label='WM Module In Node')

        sp_add_matrix = (vocab['ADD'].get_convolution_matrix() *
                         (0.25 + 0.25 / cfg.mb_decaybuf_input_scale))
        # sp_add_matrix = (vocab['ADD'].get_convolution_matrix() * 0.5)

        self.num0_bias_node = nengo.Node(vocab.parse('POS1*ZER').v)

        self.gate_sig_bias = cfg.make_thresh_ens_net()  # Bias the -1.5 neg_atn
        self.cnt_gate_sig = cfg.make_thresh_ens_net(0.5)

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.sel_mb1_in = cfg.make_selector(3, default_sel=0)
        nengo.Connection(self.mem_in, self.sel_mb1_in.input0, synapse=None)
        nengo.Connection(self.num0_bias_node, self.sel_mb1_in.input2,
                         synapse=None)

        self.mb1 = nengo.Node(size_in=cfg.sp_dim, label='MB1 Out Node')
        self.mb1_gate = nengo.Node(size_in=1, label='MB1 Gate Node')
        self.mb1_reset = nengo.Node(size_in=1, label='MB1 Reset Node')

        self.mb1a = cfg.make_mem_block(vocab=vocab, label='MB1A (Rehearsal)',
                                       reset_key=0)
        self.mb1b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB1B (Decay)')

        nengo.Connection(self.sel_mb1_in.output, self.mb1a.input)
        nengo.Connection(self.sel_mb1_in.output, self.mb1b.input,
                         transform=cfg.mb_decaybuf_input_scale)

        # Feedback gating ensembles. NOTE: Needs thresholded input as gate
        self.mb1a_fdbk_gate = cfg.make_ens_array_gate()
        self.mb1b_fdbk_gate = cfg.make_ens_array_gate()

        nengo.Connection(self.mb1a.output, self.mb1a_fdbk_gate.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb1a_fdbk_gate.output, self.mb1a.input)
        nengo.Connection(self.mb1b.output, self.mb1b_fdbk_gate.input)
        nengo.Connection(self.mb1b_fdbk_gate.output, self.mb1b.input)

        nengo.Connection(self.mb1a.output, self.mb1, synapse=None)
        nengo.Connection(self.mb1b.output, self.mb1, synapse=None)

        nengo.Connection(self.mb1_gate, self.mb1a.gate, synapse=None)
        nengo.Connection(self.mb1_gate, self.mb1b.gate, synapse=None)

        nengo.Connection(self.mb1_reset, self.mb1a.reset, synapse=None)
        nengo.Connection(self.mb1_reset, self.mb1b.reset, synapse=None)

        nengo.Connection(self.mb1, self.sel_mb1_in.input1,
                         transform=sp_add_matrix)
        nengo.Connection(self.gate_sig_bias.output, self.mb1_gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb1_gate,
                         transform=1.5)

        # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
        self.sel_mb2_in = cfg.make_selector(3, default_sel=0)
        nengo.Connection(self.mem_in, self.sel_mb2_in.input0, synapse=None)
        nengo.Connection(self.num0_bias_node, self.sel_mb2_in.input2,
                         synapse=None)

        self.mb2 = nengo.Node(size_in=cfg.sp_dim, label='MB2 Out Node')
        self.mb2_gate = nengo.Node(size_in=1, label='MB2 Gate Node')
        self.mb2_reset = nengo.Node(size_in=1, label='MB2 Reset Node')

        self.mb2a = cfg.make_mem_block(vocab=vocab, label='MB2A (Rehearsal)',
                                       reset_key=0)
        self.mb2b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB2B (Decay)')
        nengo.Connection(self.sel_mb2_in.output, self.mb2a.input)
        nengo.Connection(self.sel_mb2_in.output, self.mb2b.input,
                         transform=cfg.mb_decaybuf_input_scale)

        # Feedback gating ensembles. NOTE: Needs thresholded input as gate
        self.mb2a_fdbk_gate = cfg.make_ens_array_gate()
        self.mb2b_fdbk_gate = cfg.make_ens_array_gate()

        nengo.Connection(self.mb2a.output, self.mb2a_fdbk_gate.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb2a_fdbk_gate.output, self.mb2a.input)
        nengo.Connection(self.mb2b.output, self.mb2b_fdbk_gate.input)
        nengo.Connection(self.mb2b_fdbk_gate.output, self.mb2b.input)

        nengo.Connection(self.mb2a.output, self.mb2, synapse=None)
        nengo.Connection(self.mb2b.output, self.mb2, synapse=None)

        nengo.Connection(self.mb2_gate, self.mb2a.gate, synapse=None)
        nengo.Connection(self.mb2_gate, self.mb2b.gate, synapse=None)

        nengo.Connection(self.mb2_reset, self.mb2a.reset, synapse=None)
        nengo.Connection(self.mb2_reset, self.mb2b.reset, synapse=None)

        nengo.Connection(self.mb2, self.sel_mb2_in.input1,
                         transform=sp_add_matrix)
        nengo.Connection(self.gate_sig_bias.output, self.mb2_gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb2_gate,
                         transform=1.5)

        # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
        self.sel_mb3_in = cfg.make_selector(3, default_sel=0)
        nengo.Connection(self.mem_in, self.sel_mb3_in.input0, synapse=None)
        nengo.Connection(self.num0_bias_node, self.sel_mb3_in.input2,
                         synapse=None)

        self.mb3 = nengo.Node(size_in=cfg.sp_dim, label='MB3 Out Node')
        self.mb3_gate = nengo.Node(size_in=1, label='MB3 Gate Node')
        self.mb3_reset = nengo.Node(size_in=1, label='MB3 Reset Node')

        self.mb3a = cfg.make_mem_block(vocab=vocab, label='MB3A (Rehearsal)',
                                       reset_key=0)
        self.mb3b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB3B (Decay)')
        nengo.Connection(self.sel_mb3_in.output, self.mb3a.input)
        nengo.Connection(self.sel_mb3_in.output, self.mb3b.input,
                         transform=cfg.mb_decaybuf_input_scale)

        # Feedback gating ensembles. NOTE: Needs thresholded input as gate
        self.mb3a_fdbk_gate = cfg.make_ens_array_gate()
        self.mb3b_fdbk_gate = cfg.make_ens_array_gate()

        nengo.Connection(self.mb3a.output, self.mb3a_fdbk_gate.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb3a_fdbk_gate.output, self.mb3a.input)
        nengo.Connection(self.mb3b.output, self.mb3b_fdbk_gate.input)
        nengo.Connection(self.mb3b_fdbk_gate.output, self.mb3b.input)

        nengo.Connection(self.mb3a.output, self.mb3, synapse=None)
        nengo.Connection(self.mb3b.output, self.mb3, synapse=None)

        nengo.Connection(self.mb3_gate, self.mb3a.gate, synapse=None)
        nengo.Connection(self.mb3_gate, self.mb3b.gate, synapse=None)

        nengo.Connection(self.mb3_reset, self.mb3a.reset, synapse=None)
        nengo.Connection(self.mb3_reset, self.mb3b.reset, synapse=None)

        nengo.Connection(self.mb3, self.sel_mb3_in.input1,
                         transform=sp_add_matrix)
        nengo.Connection(self.gate_sig_bias.output, self.mb3_gate,
                         transform=2.25)
        nengo.Connection(self.cnt_gate_sig.output, self.mb3_gate,
                         transform=1.5)

        # Memory block Ave (MBAve)
        self.mb_ave = cfg.make_mem_block(vocab=vocab, label='MBAve',
                                         reset_key=0)

        self.mbave_in = nengo.Node(size_in=cfg.sp_dim, label='MBAve In Node')
        self.mbave = self.mb_ave.output
        self.mbave_gate = self.mb_ave.gate
        self.mbave_reset = self.mb_ave.reset

        # Input to mb ave = alpha * input
        nengo.Connection(self.mbave_in, self.mb_ave.input, synapse=None,
                         transform=cfg.trans_ave_scale)

        # Feedback from mb ave to mb ave = 1 - alpha
        nengo.Connection(self.mb_ave.output, self.mb_ave.input,
                         transform=(1 - cfg.trans_ave_scale))

        # Initial input to mb ave = input * (1 - alpha)
        # - So that mb ave is initialized with full input when empty
        self.mbave_in_init = cfg.make_ens_array_gate()
        nengo.Connection(self.mbave_in, self.mbave_in_init.input)
        nengo.Connection(self.mbave_in_init.output, self.mb_ave.input,
                         transform=(1 - cfg.trans_ave_scale))

        # Output norm calculation for mb ave (to shut off init input to mbave)
        self.mb_ave.mem2.mem.add_output('squared', lambda x: x * x)
        self.mbave_norm = cfg.make_thresh_ens_net()
        nengo.Connection(self.mb_ave.mem2.mem.squared, self.mbave_norm.input,
                         transform=np.ones((1, cfg.sp_dim)))
        nengo.Connection(self.mbave_norm.output, self.mbave_in_init.gate)

        # Define network inputs and outputs
        self.input = self.mem_in

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            # ###### MB1 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb1_gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb1_gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb1_reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MB2 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb2_gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb2_gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb2_reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MB3 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb3_gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb3_gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb3_reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            # ###### MBAve ########
            nengo.Connection(p_net.vis.am_utilities[ave_mb_gate_sp_inds],
                             self.mbave_gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(ave_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mbave_gate, transform=-1.5, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[ave_mb_rst_sp_inds],
                             self.mbave_reset,
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
                             self.mb1_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_gate_inds)])

            mb1_no_reset_strs = ['QAP', 'QAK', 'TRANS1', 'CNT0', 'CNT1']
            mb1_no_reset_inds = strs_to_inds(mb1_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb1_no_reset_inds],
                             self.mb1_reset,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_reset_inds)])

            mb1_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb1_no_gate_inds = strs_to_inds(mb1_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb1_no_gate_inds],
                             self.mb1_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_gate_inds)])

            mb1_sel_1_strs = ['CNT1']  # Use *ONE connection for CNT1 state
            mb1_sel_1_inds = strs_to_inds(mb1_sel_1_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.sel_mb1_in.sel1)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb1a_fdbk_gate.gate)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb1b_fdbk_gate.gate)

            # ###### MB2 ########
            mb2_no_gate_strs = ['TRANS0', 'TRANS2', 'CNT1']
            mb2_no_gate_inds = strs_to_inds(mb2_no_gate_strs,
                                            ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb2_no_gate_inds],
                             self.mb2_gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_gate_inds)])

            mb2_no_reset_strs = ['QAP', 'QAK', 'TRANS1', 'TRANS2', 'CNT1']
            mb2_no_reset_inds = strs_to_inds(mb2_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb2_no_reset_inds],
                             self.mb2_reset,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_reset_inds)])

            mb2_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb2_no_gate_inds = strs_to_inds(mb2_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb2_no_gate_inds],
                             self.mb2_gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb2_no_gate_inds)])

            # ###### MB3 ########
            mb3_no_gate_strs = ['QAP', 'QAK', 'TRANS0', 'TRANS1']
            mb3_no_gate_inds = strs_to_inds(mb3_no_gate_strs,
                                            ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_no_gate_inds],
                             self.mb3_gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_gate_inds)])

            mb3_no_reset_strs = ['CNT1']
            mb3_no_reset_inds = strs_to_inds(mb3_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_no_reset_inds],
                             self.mb3_reset,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_reset_inds)])

            mb3_no_gate_strs = ['X']  # Don't store in mb when in task=X
            mb3_no_gate_inds = strs_to_inds(mb3_no_gate_strs,
                                            ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mb3_no_gate_inds],
                             self.mb3_gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mb3_no_gate_inds)])

            mb3_sel_1_strs = ['CNT1']  # Use *ONE connection for CNT1 state
            mb3_sel_1_inds = strs_to_inds(mb3_sel_1_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_sel_1_inds],
                             self.sel_mb3_in.sel1)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb3a_fdbk_gate.gate)
            nengo.Connection(ps_state_mb_utils[mb1_sel_1_inds],
                             self.mb3b_fdbk_gate.gate)

            mb3_sel_2_strs = ['CNT0']  # Use POS1*ONE connection for CNT0 state
            mb3_sel_2_inds = strs_to_inds(mb3_sel_2_strs,
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mb3_sel_2_inds],
                             self.sel_mb3_in.sel2)

            # ###### MBAVe ########
            mbave_no_gate_strs = ['QAP', 'QAK', 'TRANS0']
            mbave_no_gate_inds = strs_to_inds(mbave_no_gate_strs,
                                              ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[mbave_no_gate_inds],
                             self.mbave_gate,
                             transform=[[-cfg.mb_gate_scale * 2] *
                                        len(mbave_no_gate_inds)])

            mbave_do_reset_strs = ['X']
            mbave_do_reset_inds = strs_to_inds(mbave_do_reset_strs,
                                               ps_task_sp_strs)
            nengo.Connection(ps_task_mb_utils[mbave_do_reset_inds],
                             self.mbave_reset,
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
            nengo.Connection(p_net.trfm.output, self.mbave_in)
        else:
            warn("WorkingMemory Module - Cannot connect from 'trfm'")

        # Set up connections from motor module (for counting task)
        if hasattr(parent_net, 'mtr'):
            nengo.Connection(parent_net.mtr.ramp_50_75.output,
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
        self.mb1_gate = nengo.Node(size_in=1, label='MB1 Gate Node')
        self.mb1_reset = nengo.Node(size_in=1, label='MB1 Reset Node')

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
