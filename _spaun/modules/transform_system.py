import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..utils import strs_to_inds
from ..vocabs import vocab, item_vocab, pos_vocab, pos1_vocab
from ..vocabs import ps_state_sp_strs, ps_dec_sp_strs
from .._spa import Compare


def invol_matrix(dim):
    result = np.eye(dim)
    return result[-np.arange(dim), :]


class TransformationSystem(Module):
    def __init__(self):
        super(TransformationSystem, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # ----- Input and output selectors ----- #
        self.select_in_a = cfg.make_selector(3)
        self.select_in_b = cfg.make_selector(6)
        self.select_out = cfg.make_selector(5)

        # ----- Normalization networks for inputs to CConv and Compare ----- #
        self.norm_a = cfg.make_norm_net()
        self.norm_b = cfg.make_norm_net()

        nengo.Connection(self.select_in_a.output, self.norm_a.input)
        nengo.Connection(self.select_in_b.output, self.norm_b.input)

        # ----- Cir conv 1 ----- #
        self.cconv1 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)

        nengo.Connection(self.norm_a.output, self.cconv1.A,
                         transform=1.25)
        nengo.Connection(self.norm_b.output, self.cconv1.B,
                         transform=1.25)
        nengo.Connection(self.cconv1.output, self.select_out.input3,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.cconv1.output, self.select_out.input4)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)
        self.frm_mbave = nengo.Node(size_in=cfg.sp_dim)

        nengo.Connection(self.frm_mb1, self.select_in_a.input0, synapse=None)
        nengo.Connection(self.frm_mb2, self.select_in_a.input1, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_a.input2, synapse=None)

        nengo.Connection(self.frm_mb2, self.select_in_b.input2, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mb3, self.select_in_b.input3, synapse=None,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.frm_mbave, self.select_in_b.input4, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_b.input5, synapse=None)

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

        nengo.Connection(self.am_p1.output, self.select_in_b.input0,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_n1.output, self.select_in_b.input1,
                         transform=invol_matrix(cfg.sp_dim))
        nengo.Connection(self.am_n2.output, self.select_out.input0)
        nengo.Connection(self.am_p2.output, self.select_out.input1)

        # ----- Compare transformation (for counting task) -----
        # TODO: Make routing to compare non-hardcoded to MB2 and MB3?
        self.compare = \
            Compare(vocab, output_no_match=True, threshold_outputs=0.5,
                    dot_product_input_magnitude=cfg.get_optimal_sp_radius(),
                    label="Compare")

        nengo.Connection(self.norm_a.output, self.compare.inputA,
                         transform=1.5)
        nengo.Connection(self.norm_b.output, self.compare.inputB,
                         transform=1.5)

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

        # ----- Set up module vocab inputs and outputs -----
        self.outputs = dict(compare=(self.compare.output, vocab))

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            # Only create this connection if we are using the LIF vision system
            if p_net.vis.mb_output.size_out == cfg.vis_dim:
                nengo.Connection(p_net.vis.mb_output, self.vis_transform.input)
        else:
            warn("TransformationSystem Module - Cannot connect from 'vis'")

        # Set up connections from vision module
        if hasattr(p_net, 'ps'):
            ps_state_mb_utils = p_net.ps.ps_state_utilities
            ps_dec_mb_utils = p_net.ps.ps_dec_utilities

            # Select CC1 A
            # - sel0 (MB1): State = QAP + QAK + TRANS1
            # - sel1 (MB2): State = TRANS2, CNT1
            # - sel2 (MB3): State = TRANS0
            in_a_sel0_inds = strs_to_inds(['QAP', 'QAK', 'TRANS1'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel0_inds],
                             self.select_in_a.sel0,
                             transform=[[1] * len(in_a_sel0_inds)])

            in_a_sel1_inds = strs_to_inds(['TRANS2', 'CNT1'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel1_inds],
                             self.select_in_a.sel1,
                             transform=[[1] * len(in_a_sel1_inds)])

            in_a_sel2_inds = strs_to_inds(['TRANS0'],
                                          ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_a_sel2_inds],
                             self.select_in_a.sel2)

            # Select CC1 B
            # - sel0 (~AM_P1): State = QAP
            # - sel1 (~AM_N1): State = QAK
            # - sel2 (~MB1): State = TRANS1; Dec = -DECI
            # - sel3 (~MB2): State = TRANS2; Dec = -DECI
            # - sel4 (MBAve): Dec = DECI
            # - sel5 (MB3): State = CNT1
            in_b_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel0_inds],
                             self.select_in_b.sel0)

            in_b_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel1_inds],
                             self.select_in_b.sel1)

            in_b_sel2 = cfg.make_thresh_ens_net()
            in_b_sel2_inds = strs_to_inds(['TRANS1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel2_inds],
                             in_b_sel2.input)
            in_b_sel2_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel2_inds],
                             in_b_sel2.input, transform=-1)
            nengo.Connection(in_b_sel2.output, self.select_in_b.sel2)

            in_b_sel3 = cfg.make_thresh_ens_net()
            in_b_sel3_inds = strs_to_inds(['TRANS2'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel3_inds],
                             in_b_sel3.input)
            in_b_sel3_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel3_inds],
                             in_b_sel3.input, transform=-1)
            nengo.Connection(in_b_sel3.output, self.select_in_b.sel3)

            in_b_sel4_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[in_b_sel4_inds],
                             self.select_in_b.sel4)

            in_b_sel5_inds = strs_to_inds(['CNT1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[in_b_sel5_inds],
                             self.select_in_b.sel5)

            # Select Output
            # - sel0 (AM_N2): State = QAP
            # - sel1 (AM_P2): State = QAK
            # - sel2 (MB1): State = TRANS0 + CNT1, Dec = -DECI
            # - sel3 (~CC1 Out): State = TRANS1 + TRANS2, Dec = -DecI
            # - sel4 (CC1 Out): Dec = DECI
            out_sel0_inds = strs_to_inds(['QAP'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel0_inds],
                             self.select_out.sel0)

            out_sel1_inds = strs_to_inds(['QAK'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel1_inds],
                             self.select_out.sel1)

            out_sel2 = cfg.make_thresh_ens_net()
            out_sel2_inds = strs_to_inds(['TRANS0', 'CNT1'], ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel2_inds], out_sel2.input,
                             transform=[[1] * len(out_sel2_inds)])
            out_sel2_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel2_inds], out_sel2.input,
                             transform=-1)
            nengo.Connection(out_sel2.output, self.select_out.sel2)

            out_sel3 = cfg.make_thresh_ens_net()
            out_sel3_inds = strs_to_inds(['TRANS1', 'TRANS2'],
                                         ps_state_sp_strs)
            nengo.Connection(ps_state_mb_utils[out_sel3_inds], out_sel3.input,
                             transform=[[1] * len(out_sel3_inds)])
            out_sel3_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel3_inds], out_sel3.input,
                             transform=-1)
            nengo.Connection(out_sel3.output, self.select_out.sel3)

            out_sel4_inds = strs_to_inds(['DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[out_sel4_inds],
                             self.select_out.sel4)

            # Disable input normalization for Dec = DECI
            dis_norm_inds = strs_to_inds(['FWD', 'DECI'], ps_dec_sp_strs)
            nengo.Connection(ps_dec_mb_utils[dis_norm_inds],
                             self.norm_a.disable,
                             transform=[[1] * len(dis_norm_inds)])
            nengo.Connection(ps_dec_mb_utils[dis_norm_inds],
                             self.norm_b.disable,
                             transform=[[1] * len(dis_norm_inds)])
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
        self.select_in_a = cfg.make_selector(2, n_ensembles=1,
                                             ens_dimensions=cfg.sp_dim,
                                             n_neurons=cfg.sp_dim)
        self.select_in_b = cfg.make_selector(5, n_ensembles=1,
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

        # ----- Compare network (for counting task) -----
        def cmp_func(x, cmp_vocab):
            vec_A = x[:cfg.sp_dim]
            vec_B = x[cfg.sp_dim:]
            if np.linalg.norm(vec_A) != 0:
                vec_A = vec_A / np.linalg.norm(vec_A)
            if np.linalg.norm(vec_B) != 0:
                vec_B = vec_B / np.linalg.norm(vec_B)
            dot_val = np.dot(vec_A, vec_B)
            conj_val = 1 - dot_val
            if dot_val > conj_val:
                return cmp_vocab.parse('MATCH').v
            else:
                return cmp_vocab.parse('NO_MATCH').v

        self.compare = \
            nengo.Node(size_in=cfg.sp_dim * 2,
                       output=lambda t, x: cmp_func(x, cmp_vocab=vocab))

        nengo.Connection(self.frm_mb2, self.compare[:cfg.sp_dim])
        nengo.Connection(self.frm_mb3, self.compare[cfg.sp_dim:])

        # ----- Output node -----
        self.output = self.frm_mb1
        self.outputs = dict(compare=(self.compare, vocab))
