import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._spa import Compare
from ..configurator import cfg
from ..vocabulator import vocab
from ..utils import invol_matrix
from .transform import Assoc_Mem_Transforms_Network


class TransformationSystem(Module):
    def __init__(self, label="Transformation Sys", seed=None,
                 add_to_container=None):
        super(TransformationSystem, self).__init__(label, seed,
                                                   add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        # ----- Input and output selectors ----- #
        self.select_in_a = cfg.make_selector(3)
        self.select_in_b = cfg.make_selector(6, represent_identity=True)
        self.select_out = cfg.make_selector(6, represent_identity=True)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mbave = nengo.Node(size_in=vocab.sp_dim)
        self.frm_action = nengo.Node(size_in=vocab.sp_dim)

        nengo.Connection(self.frm_mb1, self.select_in_a.input0, synapse=None)
        nengo.Connection(self.frm_mb2, self.select_in_a.input1, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_a.input2, synapse=None)

        nengo.Connection(self.frm_mb2, self.select_in_b.input2, synapse=None,
                         transform=invol_matrix(vocab.sp_dim))
        nengo.Connection(self.frm_mb3, self.select_in_b.input3, synapse=None,
                         transform=invol_matrix(vocab.sp_dim))
        nengo.Connection(self.frm_mbave, self.select_in_b.input4, synapse=None)
        nengo.Connection(self.frm_mb3, self.select_in_b.input5, synapse=None)

        nengo.Connection(self.frm_mb1, self.select_out.input2)

        # ----- Normalization networks for inputs to CConv and Compare ----- #
        self.norm_a = cfg.make_norm_net()  # Note: Normalization is disabled
        self.norm_b = cfg.make_norm_net()  # for certain dec phases - see below

        nengo.Connection(self.select_in_a.output, self.norm_a.input)
        nengo.Connection(self.select_in_b.output, self.norm_b.input)

        # ----- Cir conv 1 ----- #
        self.cconv1 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)

        nengo.Connection(self.norm_a.output, self.cconv1.A)
        nengo.Connection(self.norm_b.output, self.cconv1.B)
        nengo.Connection(self.cconv1.output, self.select_out.input3,
                         transform=invol_matrix(vocab.sp_dim))
        nengo.Connection(self.cconv1.output, self.select_out.input4)

        # ----- Assoc memory transforms (for QA task and learning) -----
        self.am_trfms = Assoc_Mem_Transforms_Network(vocab.item,
                                                     vocab.pos, vocab.pos1,
                                                     vocab.max_enum_list_pos,
                                                     vocab.ps_action_learn)

        nengo.Connection(self.frm_mb1, self.am_trfms.frm_mb1, synapse=None)
        nengo.Connection(self.frm_mb2, self.am_trfms.frm_mb2, synapse=None)
        nengo.Connection(self.frm_mb3, self.am_trfms.frm_mb3, synapse=None)
        nengo.Connection(self.frm_action, self.am_trfms.frm_action,
                         synapse=None)

        nengo.Connection(self.cconv1.output, self.am_trfms.frm_cconv)

        nengo.Connection(self.am_trfms.pos1_to_pos, self.select_in_b.input0,
                         transform=invol_matrix(vocab.sp_dim))
        nengo.Connection(self.am_trfms.pos1_to_num, self.select_in_b.input1,
                         transform=invol_matrix(vocab.sp_dim))
        nengo.Connection(self.am_trfms.num_to_pos1, self.select_out.input0)
        nengo.Connection(self.am_trfms.pos_to_pos1, self.select_out.input1)

        nengo.Connection(self.am_trfms.action_out, self.select_out.input5)

        # ----- Compare transformation (for counting task) -----
        self.compare = \
            Compare(vocab.main, output_no_match=True, threshold_outputs=0.5,
                    dot_product_input_magnitude=cfg.get_optimal_sp_radius(),
                    label="Compare")

        nengo.Connection(self.norm_a.output, self.compare.inputA,
                         transform=1.5)
        nengo.Connection(self.norm_b.output, self.compare.inputB,
                         transform=1.5)

        # ----- Output node -----
        self.output = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(self.select_out.output, self.output, synapse=None)

        # ----- Set up module vocab inputs and outputs -----
        self.outputs = dict(compare=(self.compare.output, vocab.main))

    @with_self
    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from ps module
        if hasattr(p_net, 'ps'):
            nengo.Connection(p_net.ps.action, self.frm_action)

            # Select IN A
            # - sel0 (MB1): State = QAP + QAK + TRANS1
            # - sel1 (MB2): State = TRANS2, CNT1
            # - sel2 (MB3): State = TRANS0
            in_a_sel0_sp_vecs = vocab.main.parse('QAP+QAK+TRANS1').v
            nengo.Connection(p_net.ps.state, self.select_in_a.sel0,
                             transform=[in_a_sel0_sp_vecs])

            in_a_sel1_sp_vecs = vocab.main.parse('TRANS2+CNT1').v
            nengo.Connection(p_net.ps.state, self.select_in_a.sel1,
                             transform=[in_a_sel1_sp_vecs])

            in_a_sel2_sp_vecs = vocab.main.parse('TRANS0').v
            nengo.Connection(p_net.ps.state, self.select_in_a.sel2,
                             transform=[in_a_sel2_sp_vecs])

            # Select IN B
            # - sel0 (~AM_P1): State = QAP
            # - sel1 (~AM_N1): State = QAK
            # - sel2 (~MB1): State = TRANS1 & Dec = -DECI
            # - sel3 (~MB2): State = TRANS2 & Dec = -DECI
            # - sel4 (MBAve): Dec = DECI
            # - sel5 (MB3): State = CNT1
            in_b_sel0_sp_vecs = vocab.main.parse('QAP').v
            nengo.Connection(p_net.ps.state, self.select_in_b.sel0,
                             transform=[in_b_sel0_sp_vecs])

            in_b_sel1_sp_vecs = vocab.main.parse('QAK').v
            nengo.Connection(p_net.ps.state, self.select_in_b.sel1,
                             transform=[in_b_sel1_sp_vecs])

            in_b_sel2_sp_vecs = vocab.main.parse('TRANS1-DECI').v
            nengo.Connection(p_net.ps.state, self.select_in_b.sel2,
                             transform=[in_b_sel2_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.select_in_b.sel2,
                             transform=[in_b_sel2_sp_vecs])

            in_b_sel3_sp_vecs = vocab.main.parse('TRANS2-DECI').v
            nengo.Connection(p_net.ps.state, self.select_in_b.sel3,
                             transform=[in_b_sel3_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.select_in_b.sel3,
                             transform=[in_b_sel3_sp_vecs])

            in_b_sel4_sp_vecs = vocab.main.parse('DECI').v
            nengo.Connection(p_net.ps.dec, self.select_in_b.sel4,
                             transform=[in_b_sel4_sp_vecs])

            in_b_sel5_sp_vecs = vocab.main.parse('CNT1').v
            nengo.Connection(p_net.ps.state, self.select_in_b.sel5,
                             transform=[in_b_sel5_sp_vecs])

            # Select Output
            # - sel0 (AM_N2): State = QAP
            # - sel1 (AM_P2): State = QAK
            # - sel2 (MB1): State = TRANS0 + CNT1 & Dec = -DECI
            # - sel3 (~CC1 Out): State = TRANS1 + TRANS2 & Dec = -DECI
            # - sel4 (CC1 Out): Dec = DECI
            # - sel5 (ACT LEARN Out): State = LEARN & Dec = -NONE
            out_sel0_sp_vecs = vocab.main.parse('QAP').v
            nengo.Connection(p_net.ps.state, self.select_out.sel0,
                             transform=[out_sel0_sp_vecs])

            out_sel1_sp_vecs = vocab.main.parse('QAK').v
            nengo.Connection(p_net.ps.state, self.select_out.sel1,
                             transform=[out_sel1_sp_vecs])

            out_sel2_sp_vecs = vocab.main.parse('TRANS0+CNT1-DECI').v
            nengo.Connection(p_net.ps.state, self.select_out.sel2,
                             transform=[out_sel2_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.select_out.sel2,
                             transform=[out_sel2_sp_vecs])

            out_sel3_sp_vecs = vocab.main.parse('TRANS1+TRANS2-DECI').v
            nengo.Connection(p_net.ps.state, self.select_out.sel3,
                             transform=[out_sel3_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.select_out.sel3,
                             transform=[out_sel3_sp_vecs])

            out_sel4_sp_vecs = vocab.main.parse('DECI').v
            nengo.Connection(p_net.ps.dec, self.select_out.sel4,
                             transform=[out_sel4_sp_vecs])

            out_sel5_sp_vecs = vocab.main.parse('LEARN-NONE').v
            nengo.Connection(p_net.ps.state, self.select_out.sel5,
                             transform=[out_sel5_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.select_out.sel5,
                             transform=[out_sel5_sp_vecs])

            # Disable input normalization for Dec = DECI + FWD + REV
            dis_norm_sp_vecs = vocab.main.parse('FWD+REV+DECI').v
            nengo.Connection(p_net.ps.dec, self.norm_a.disable,
                             transform=[dis_norm_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.norm_b.disable,
                             transform=[dis_norm_sp_vecs])
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
                                             ens_dimensions=vocab.sp_dim,
                                             n_neurons=vocab.sp_dim)
        self.select_in_b = cfg.make_selector(5, n_ensembles=1,
                                             ens_dimensions=vocab.sp_dim,
                                             n_neurons=vocab.sp_dim)
        self.select_out = cfg.make_selector(4, n_ensembles=1,
                                            ens_dimensions=vocab.sp_dim,
                                            n_neurons=vocab.sp_dim)

        # ----- Mem inputs and outputs ----- #
        self.frm_mb1 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mb2 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mb3 = nengo.Node(size_in=vocab.sp_dim)
        self.frm_mbave = nengo.Node(size_in=vocab.sp_dim)

        # ----- Compare network (for counting task) -----
        def cmp_func(x, cmp_vocab):
            vec_A = x[:vocab.sp_dim]
            vec_B = x[vocab.sp_dim:]
            if np.linalg.norm(vec_A) != 0:
                vec_A = vec_A / np.linalg.norm(vec_A)
            if np.linalg.norm(vec_B) != 0:
                vec_B = vec_B / np.linalg.norm(vec_B)
            dot_val = np.dot(vec_A, vec_B)
            conj_val = 1 - dot_val
            if dot_val > conj_val:
                return vocab.cmp.parse('MATCH').v
            else:
                return vocab.cmp.parse('NO_MATCH').v

        self.compare = \
            nengo.Node(size_in=vocab.sp_dim * 2,
                       output=lambda t, x: cmp_func(x, cmp_vocab=vocab))

        nengo.Connection(self.frm_mb2, self.compare[:vocab.sp_dim])
        nengo.Connection(self.frm_mb3, self.compare[vocab.sp_dim:])

        # ----- Output node -----
        self.output = self.frm_mb1
        self.outputs = dict(compare=(self.compare, vocab))
