from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..utils import strs_to_inds
from ..vocabs import vocab, ps_state_sp_strs, ps_task_sp_strs
from ..vocabs import item_mb_gate_sp_inds, item_mb_rst_sp_inds
from ..vocabs import ave_mb_gate_sp_inds, ave_mb_rst_sp_inds


class WorkingMemory(Module):
    def __init__(self):
        super(WorkingMemory, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim, label='WM Module In Node')

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1 = nengo.Node(size_in=cfg.sp_dim, label='MB1 In Node')
        self.mb1_gate = nengo.Node(size_in=1, label='MB1 Gate Node')
        self.mb1_reset = nengo.Node(size_in=1, label='MB1 Reset Node')

        self.mb1a = cfg.make_mem_block(vocab=vocab, label='MB1A (Rehearsal)',
                                       reset_key=0)
        self.mb1b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB1B (Decay)')
        nengo.Connection(self.mem_in, self.mb1a.input, synapse=None)
        nengo.Connection(self.mem_in, self.mb1b.input,
                         transform=cfg.mb_decaybuf_input_scale, synapse=None)
        nengo.Connection(self.mb1a.output, self.mb1a.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb1b.output, self.mb1b.input)
        nengo.Connection(self.mb1a.output, self.mb1, synapse=None)
        nengo.Connection(self.mb1b.output, self.mb1, synapse=None)

        nengo.Connection(self.mb1_gate, self.mb1a.gate, synapse=None)
        nengo.Connection(self.mb1_gate, self.mb1b.gate, synapse=None)

        nengo.Connection(self.mb1_reset, self.mb1a.reset, synapse=None)
        nengo.Connection(self.mb1_reset, self.mb1b.reset, synapse=None)

        # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
        self.mb2 = nengo.Node(size_in=cfg.sp_dim, label='MB2 In Node')
        self.mb2_gate = nengo.Node(size_in=1, label='MB2 Gate Node')
        self.mb2_reset = nengo.Node(size_in=1, label='MB2 Reset Node')

        self.mb2a = cfg.make_mem_block(vocab=vocab, label='MB2A (Rehearsal)',
                                       reset_key=0)
        self.mb2b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB2B (Decay)')
        nengo.Connection(self.mem_in, self.mb2a.input, synapse=None)
        nengo.Connection(self.mem_in, self.mb2b.input,
                         transform=cfg.mb_decaybuf_input_scale, synapse=None)
        nengo.Connection(self.mb2a.output, self.mb2a.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb2b.output, self.mb2b.input)
        nengo.Connection(self.mb2a.output, self.mb2, synapse=None)
        nengo.Connection(self.mb2b.output, self.mb2, synapse=None)

        nengo.Connection(self.mb2_gate, self.mb2a.gate, synapse=None)
        nengo.Connection(self.mb2_gate, self.mb2b.gate, synapse=None)

        nengo.Connection(self.mb2_reset, self.mb2a.reset, synapse=None)
        nengo.Connection(self.mb2_reset, self.mb2b.reset, synapse=None)

        # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
        self.mb3 = nengo.Node(size_in=cfg.sp_dim, label='MB3 In Node')
        self.mb3_gate = nengo.Node(size_in=1, label='MB3 Gate Node')
        self.mb3_reset = nengo.Node(size_in=1, label='MB3 Reset Node')

        self.mb3a = cfg.make_mem_block(vocab=vocab, label='MB3A (Rehearsal)',
                                       reset_key=0)
        self.mb3b = cfg.make_mem_block(vocab=vocab,
                                       fdbk_transform=cfg.mb_decay_val,
                                       reset_key=0,
                                       label='MB3B (Decay)')
        nengo.Connection(self.mem_in, self.mb3a.input, synapse=None)
        nengo.Connection(self.mem_in, self.mb3b.input,
                         transform=cfg.mb_decaybuf_input_scale, synapse=None)
        nengo.Connection(self.mb1a.output, self.mb3a.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb3b.output, self.mb3b.input)
        nengo.Connection(self.mb3a.output, self.mb3, synapse=None)
        nengo.Connection(self.mb3b.output, self.mb3, synapse=None)

        nengo.Connection(self.mb3_gate, self.mb3a.gate, synapse=None)
        nengo.Connection(self.mb3_gate, self.mb3b.gate, synapse=None)

        nengo.Connection(self.mb3_reset, self.mb3a.reset, synapse=None)
        nengo.Connection(self.mb3_reset, self.mb3b.reset, synapse=None)

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

        # Define network inputs and outputs
        # ## TODO: Fix this! (update to include selector and what not)
        self.input = self.mem_in
        self.output = self.mb1  # Todo: Remove me

        # ## TODO: add selector (to direct info flow to MB units) & other
        #         MB units (mb2, mb3, mbcnt, mbave)

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
                             self.mb1_gate, transform=-1, synapse=0.01)

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
                             self.mb2_gate, transform=-1, synapse=0.01)

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
                             self.mb3_gate, transform=-1, synapse=0.01)

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
                             self.mbave_gate, transform=-1, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[ave_mb_rst_sp_inds],
                             self.mbave_reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(ave_mb_rst_sp_inds)])
        else:
            warn("WorkingMemory Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            ps_state_mb_thresh = p_net.ps.ps_state_mb.mem2.mem.output
            ps_task_mb_thresh = p_net.ps.ps_task_mb.mem2.mem.output

            # ###### MB1 ########
            mb1_no_gate_strs = ['QAP', 'QAN', 'TRANS1', 'TRANS2']
            mb1_no_gate_inds = strs_to_inds(mb1_no_gate_strs, ps_state_sp_strs)

            nengo.Connection(ps_state_mb_thresh[mb1_no_gate_inds],
                             self.mb1_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_gate_inds)])

            mb1_no_reset_strs = ['QAP', 'QAN', 'TRANS1', 'TRANS0']
            mb1_no_reset_inds = strs_to_inds(mb1_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[mb1_no_reset_inds],
                             self.mb1_reset,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb1_no_reset_inds)])

            # ###### MB2 ########
            mb2_no_gate_strs = ['TRANS0', 'TRANS2']
            mb2_no_gate_inds = strs_to_inds(mb2_no_gate_strs, ps_state_sp_strs)

            nengo.Connection(ps_state_mb_thresh[mb2_no_gate_inds],
                             self.mb2_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb2_no_gate_inds)])

            mb2_no_reset_strs = ['QAP', 'QAN', 'TRANS1', 'TRANS2']
            mb2_no_reset_inds = strs_to_inds(mb2_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[mb2_no_reset_inds],
                             self.mb2_reset,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb2_no_reset_inds)])

            # ###### MB3 ########
            mb3_no_gate_strs = ['QAP', 'QAN', 'TRANS0', 'TRANS1']
            mb3_no_gate_inds = strs_to_inds(mb3_no_gate_strs, ps_state_sp_strs)

            nengo.Connection(ps_state_mb_thresh[mb3_no_gate_inds],
                             self.mb3_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb3_no_gate_inds)])

            mb3_no_reset_strs = ['TRANS2']
            mb3_no_reset_inds = strs_to_inds(mb3_no_reset_strs,
                                             ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[mb3_no_reset_inds],
                             self.mb3_reset,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mb3_no_reset_inds)])

            # ###### MBAVe ########
            mbave_no_gate_strs = ['QAP', 'QAN', 'TRANS0']
            mbave_no_gate_inds = strs_to_inds(mbave_no_gate_strs,
                                              ps_state_sp_strs)

            nengo.Connection(ps_state_mb_thresh[mbave_no_gate_inds],
                             self.mbave_gate,
                             transform=[[-cfg.mb_gate_scale] *
                                        len(mbave_no_gate_inds)])

            mbave_do_reset_strs = ['X']
            mbave_do_reset_inds = strs_to_inds(mbave_do_reset_strs,
                                               ps_task_sp_strs)
            nengo.Connection(ps_task_mb_thresh[mbave_do_reset_inds],
                             self.mbave_reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(mbave_do_reset_inds)])
        else:
            warn("WorkingMemory Module - Cannot connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.enc_output, self.mem_in)
        else:
            warn("WorkingMemory Module - Cannot connect from 'enc'")
