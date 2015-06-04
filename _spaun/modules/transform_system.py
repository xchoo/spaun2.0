import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..utils import strs_to_inds
from ..vocabs import item_vocab, pos_vocab, pos1_vocab
from ..vocabs import ps_state_sp_strs, ps_task_sp_strs


def invol_matrix(dim):
    result = np.eye(dim)
    return result[-np.arange(dim), :]


class TransformationSystem(Module):
    def __init__(self):
        super(TransformationSystem, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # ----- Cir conv 1 ----- #
        self.cconv1 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)

        self.select_cc1a = cfg.make_selector(2)
        self.select_cc1b = cfg.make_selector(5)
        self.select_out = cfg.make_selector(4)

        nengo.Connection(self.select_cc1a.output, self.cconv1.A)
        nengo.Connection(self.select_cc1b.output, self.cconv1.B)
        nengo.Connection(self.cconv1.output, self.select_out.input2)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mbave = nengo.Node(size_in=cfg.sp_dim)

        nengo.Connection(self.frm_mb1, self.select_cc1a.input0, synapse=None)
        nengo.Connection(self.frm_mb2, self.select_cc1a.input1, synapse=None)

        nengo.Connection(self.frm_mb2, self.select_cc1b.input2, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mb3, self.select_cc1b.input3, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mbave, self.select_cc1b.input4, synapse=None)

        nengo.Connection(self.frm_mb1, self.select_out.input2)

        # ----- Assoc memory transforms (for QA task) -----
        # TODO: Make routing to these non-hardcoded to MB2?
        self.am_p1 = cfg.make_assoc_mem(
            pos1_vocab.vectors[1:cfg.max_enum_list_pos + 1, :],
            pos_vocab.vectors)
        self.am_n1 = cfg.make_assoc_mem(pos1_vocab.vectors, item_vocab.vectors)

        self.am_p2 = cfg.make_assoc_mem(
            pos_vocab.vectors,
            pos1_vocab.vectors[1:cfg.max_enum_list_pos + 1, :])
        self.am_n2 = cfg.make_assoc_mem(item_vocab.vectors, pos1_vocab.vectors)

        nengo.Connection(self.frm_mb2, self.am_p1.input, synapse=None)
        nengo.Connection(self.frm_mb2, self.am_n1.input, synapse=None)
        nengo.Connection(self.cconv1.output, self.am_p2.input)
        nengo.Connection(self.cconv1.output, self.am_n2.input)

        nengo.Connection(self.am_p1.output, self.select_cc1b.input0,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_n1.output, self.select_cc1b.input1,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_n2.output, self.select_out.input0)
        nengo.Connection(self.am_p2.output, self.select_out.input1)

        # ----- Visual transformation (for Copy Draw task) -----
        from ..vision.lif_vision import am_threshold, am_vis_sps
        from ..vocabs import mtr_vocab
        self.vis_transform = \
            cfg.make_assoc_mem(am_vis_sps[:len(mtr_vocab.keys), :],
                               mtr_vocab.vectors, threshold=am_threshold,
                               inhibitable=True, inhibit_scale=3)

        # ----- Output node -----
        self.output = nengo.Node(size_in=cfg.sp_dim)

        nengo.Connection(self.select_out.output, self.output, synapse=None)

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            if p_net.vis.mb_output.size_out == cfg.vis_dim:
                nengo.Connection(p_net.vis.mb_output, self.vis_transform.input)
        else:
            warn("TransformationSystem Module - Cannot connect from 'vis'")

        # Set up connections from vision module
        if hasattr(p_net, 'ps'):
            ps_state_mb_thresh = p_net.ps.ps_state_mb.mem2.mem.thresh
            ps_task_mb_thresh = p_net.ps.ps_task_mb.mem2.mem.thresh

            # Select CC1 A
            # - sel0 (MB1): State = QAP + QAK + TRANS1
            # - sel1 (MB2): State = TRANS2
            cc1a_sel0_inds = strs_to_inds(['QAP', 'QAK', 'TRANS1'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1a_sel0_inds],
                             self.select_cc1a.sel0,
                             transform=[[1] * len(cc1a_sel0_inds)])

            cc1a_sel1_inds = strs_to_inds(['TRANS2'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1a_sel1_inds],
                             self.select_cc1a.sel1)

            # Select CC1 B
            # - sel0 (~AM_P1): State = QAP
            # - sel1 (~AM_N1): State = QAK
            # - sel2 (~MB2): State = TRANS1; Task = -DECI
            # - sel3 (~MB3): State = TRANS2; Task = -DECI
            # - sel4 (MBAve): Task = DECI
            cc1b_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1b_sel0_inds],
                             self.select_cc1b.sel0)

            cc1b_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1b_sel1_inds],
                             self.select_cc1b.sel1)

            cc1b_sel2_inds = strs_to_inds(['TRANS1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1b_sel2_inds],
                             self.select_cc1b.sel2)
            cc1b_sel2_inds = strs_to_inds(['DECI'], ps_task_sp_strs)
            nengo.Connection(ps_task_mb_thresh[cc1b_sel2_inds],
                             self.select_cc1b.sel2, transform=-1)

            cc1b_sel3_inds = strs_to_inds(['TRANS2'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[cc1b_sel3_inds],
                             self.select_cc1b.sel3)
            cc1b_sel3_inds = strs_to_inds(['DECI'], ps_task_sp_strs)
            nengo.Connection(ps_task_mb_thresh[cc1b_sel3_inds],
                             self.select_cc1b.sel3, transform=-1)

            cc1b_sel4_inds = strs_to_inds(['DECI'], ps_task_sp_strs)
            nengo.Connection(ps_task_mb_thresh[cc1b_sel4_inds],
                             self.select_cc1b.sel4)

            # Select Output
            # - sel0 (AM_N2): State = QAP
            # - sel1 (AM_P2): State = QAK
            # - sel2 (MB1): State = TRANS0 + CNT
            # - sel3 (CC1 Out): State = TRANS1 + TRANS2
            out_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[out_sel0_inds],
                             self.select_out.sel0)

            out_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[out_sel1_inds],
                             self.select_out.sel1)

            out_sel2_inds = strs_to_inds(['TRANS0', 'CNT'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[out_sel2_inds],
                             self.select_out.sel2,
                             transform=[[1] * len(out_sel2_inds)])

            out_sel3_inds = strs_to_inds(['TRANS1', 'TRANS2'],
                                         ps_state_sp_strs)
            nengo.Connection(ps_state_mb_thresh[out_sel3_inds],
                             self.select_out.sel3,
                             transform=[[1] * len(out_sel3_inds)])
        else:
            warn("TransformationSystem Module - Cannot connect from 'ps'")

        # Set up connections from memory module
        if hasattr(p_net, 'mem'):
            nengo.Connection(p_net.mem.mb1, self.frm_mb1)
            nengo.Connection(p_net.mem.mb2, self.frm_mb2)
            nengo.Connection(p_net.mem.mb3, self.frm_mb3)
            nengo.Connection(p_net.mem.mbave, self.frm_mbave)
        else:
            warn("TransformationSystem Module - Cannot connect from 'mem'")


class TransformationSystemDummy(TransformationSystem):
    def __init__(self):
        super(TransformationSystemDummy, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        self.select_cc1a = cfg.make_selector(2, n_ensembles=1,
                                             ens_dimensions=cfg.sp_dim,
                                             n_neurons=cfg.sp_dim)
        self.select_cc1b = cfg.make_selector(5, n_ensembles=1,
                                             ens_dimensions=cfg.sp_dim,
                                             n_neurons=cfg.sp_dim)
        self.select_out = cfg.make_selector(4, n_ensembles=1,
                                            ens_dimensions=cfg.sp_dim,
                                            n_neurons=cfg.sp_dim)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mbave = nengo.Node(size_in=cfg.sp_dim)

        # ----- Visual transformation (for Copy Draw task) -----
        from ..vision.lif_vision import am_threshold, am_vis_sps
        from ..vocabs import mtr_vocab
        self.vis_transform = \
            cfg.make_assoc_mem(am_vis_sps[:len(mtr_vocab.keys), :],
                               mtr_vocab.vectors, threshold=am_threshold,
                               inhibitable=True, inhibit_scale=3)

        # ----- Output node -----
        self.output = self.frm_mb1
