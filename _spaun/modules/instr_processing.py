from warnings import warn
import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from .instr import PS_Sig_Gen, Data_Sig_Gen, Set_Pos_Inc_Net


# ### DEBUG ###
def cleanup_func_wta(t, x, vectors):
    return vectors[np.argmax(np.dot(x, vectors.T)), :]


def cleanup_func(t, x, vectors):
    return np.sum(vectors[np.dot(x, vectors.T) > 0.5, :], axis=0)
# ### DEBUG ###


class InstructionProcessingSystem(Module):
    def __init__(self, label="Instr Processing Sys", seed=None,
                 add_to_container=None):
        super(InstructionProcessingSystem, self).__init__(label, seed,
                                                          add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(1)

        # ------------------------ INPUT NODES --------------------------------
        self.instr_input = nengo.Node(size_in=vocab.sp_dim)
        self.vis_input = nengo.Node(size_in=vocab.sp_dim)
        self.task_input = nengo.Node(size_in=vocab.sp_dim)
        self.state_input = nengo.Node(size_in=vocab.sp_dim)
        self.dec_input = nengo.Node(size_in=vocab.sp_dim)
        # self.vis_input = nengo.Node(size_in=vocab.sp_dim, output=lambda t, x,
        #                             vectors=vocab.vis_main.vectors:
        #                             cleanup_func_wta(t, x, vectors))
        # self.task_input = nengo.Node(size_in=vocab.sp_dim, output=lambda t, x,
        #                              vectors=vocab.ps_task.vectors:
        #                              cleanup_func(t, x, vectors))
        # self.state_input = nengo.Node(size_in=vocab.sp_dim, output=lambda t, x,
        #                               vectors=vocab.ps_state.vectors:
        #                               cleanup_func_wta(t, x, vectors))

        # ----------- INSTRUCTION SP INPUT + SEMI NORMALIZATION ---------------
        instr_ea_subdim = min(16, vocab.sp_dim)
        self.instr_ea = cfg.make_ens_array(
            n_neurons=cfg.n_neurons_ens * instr_ea_subdim,
            n_ensembles=vocab.sp_dim / instr_ea_subdim,
            ens_dimensions=instr_ea_subdim,
            radius=cfg.get_optimal_sp_radius(vocab.sp_dim, instr_ea_subdim))
        nengo.Connection(self.instr_input, self.instr_ea.input, synapse=None)

        # ------------- INSTRUCTION INPUT CLEANUP NETWORKS --------------------
        # Associative memory for visual information
        vis_am = cfg.make_assoc_mem(vocab.vis_main.vectors)
        nengo.Connection(self.vis_input, vis_am.input)

        # Ignore some visual inputs
        ignored_vis_sp = vocab.vis_main.parse('A+M+QM').v
        nengo.Connection(bias_node, vis_am.input,
                         transform=-ignored_vis_sp[:, None])

        # Associative memory for task information
        task_am = cfg.make_assoc_mem(vocab.ps_task.vectors,
                                     wta_inhibit_scale=None)
        nengo.Connection(self.task_input, task_am.input)

        # Associative memory for state information
        state_am = cfg.make_assoc_mem(vocab.ps_state.vectors)
        nengo.Connection(self.state_input, state_am.input)

        # ------------ INSTRUCTION POSITION CIRCONV NETWORK -------------------
        instr_pos_cconv = cfg.make_cir_conv(
            invert_b=True, input_magnitude=cfg.instr_cconv_radius)
        cfg.make_inhibitable(instr_pos_cconv, inhib_scale=5)
        # instr_pos_cconv = nengo.Network()
        # with instr_pos_cconv as net:
        #     from nengo.networks.circularconvolution import circconv
        #     net.A = nengo.Node(size_in=vocab.sp_dim)
        #     net.B = nengo.Node(size_in=vocab.sp_dim)
        #     net.output = nengo.Node(size_in=vocab.sp_dim * 2,
        #                             output=lambda t, x:
        #                             circconv(x[:vocab.sp_dim],
        #                                      x[vocab.sp_dim:], invert_b=True))
        #     nengo.Connection(net.A, net.output[:vocab.sp_dim], synapse=None)
        #     nengo.Connection(net.B, net.output[vocab.sp_dim:], synapse=None)

        # Input A to pos cconv is the instruction sp
        # -- Note: ens array seems to 'normalize' sp to about 2.5 in mag
        nengo.Connection(self.instr_ea.output, instr_pos_cconv.A,
                         transform=cfg.instr_cconv_radius / 2.5)

        # Input B to pos cconv is the current system state
        instr_voc = vocab.instr
        antT = vocab.perm_ant
        inv_conT = vocab.perm_con_inv

        nengo.Connection(
            vis_am.output, instr_pos_cconv.B,
            transform=instr_voc.parse('VIS').get_convolution_matrix()[antT])
        nengo.Connection(
            task_am.output, instr_pos_cconv.B,
            transform=instr_voc.parse('TASK').get_convolution_matrix()[antT])
        nengo.Connection(
            state_am.output, instr_pos_cconv.B,
            transform=instr_voc.parse('STATE').get_convolution_matrix()[antT])

        # ----------- SEQUENTIAL INSTRUCTION POSITION INC NETWORK -------------
        # Position increment network for sequential instructions
        # Note; There is no position 0 instruction. Starts at position 1.
        self.pos_inc = Set_Pos_Inc_Net(vocab.pos, vocab.main.parse('0').v,
                                       vocab.inc_sp, vocab.item_1_index,
                                       threshold_gate_in=True)
        nengo.Connection(
            self.pos_inc.output, instr_pos_cconv.B,
            transform=instr_voc.parse('1').get_convolution_matrix()[antT])

        self.pos_inc_init = cfg.make_thresh_ens_net()
        nengo.Connection(self.pos_inc_init.output, self.pos_inc.gate)

        # Only enable the pos_inc num_2_pos am in pos_inc_init state. This is
        # to ignore other non-init number inputs to the system
        self.pos_inc_inhibit = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, self.pos_inc_inhibit.input)
        nengo.Connection(self.pos_inc_init.output, self.pos_inc_inhibit.input,
                         transform=-1)
        nengo.Connection(self.pos_inc_inhibit.output, self.pos_inc.inhibit_am)

        # ----------- INSTRUCTION POSITION CLEANUP NETWORK --------------------
        pos_am = cfg.make_assoc_mem(vocab.pos.vectors, threshold=0.35,
                                    wta_inhibit_scale=1.1)
        nengo.Connection(instr_pos_cconv.output, pos_am.input, synapse=0.01)

        # Signal when no position is chosen from the pos AM
        no_pos_chosen = cfg.make_thresh_ens_net(0.5)
        nengo.Connection(
            pos_am.output, no_pos_chosen.input,
            transform=-np.sum(vocab.pos.vectors, axis=0)[:, None].T)
        nengo.Connection(bias_node, no_pos_chosen.input)

        # ----------- INSTRUCTION CONSEQUENCE CIRCONV NETWORK -----------------
        instr_cons_cconv = cfg.make_cir_conv(
            invert_b=True, input_magnitude=cfg.instr_cconv_radius)
        nengo.Connection(self.instr_input, instr_cons_cconv.A)
        nengo.Connection(pos_am.output, instr_cons_cconv.B)

        # -------------------- INSTRUCTION OUTPUTS ----------------------------
        # Instruction DATA output
        data_sig_gen = Data_Sig_Gen(vocab.main, 'DATA')
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], data_sig_gen.input,
            transform=(instr_voc.parse('~DATA').get_convolution_matrix() *
                       cfg.instr_out_gain), synapse=0.01)
        nengo.Connection(no_pos_chosen.output, data_sig_gen.gate_sig_in,
                         transform=-80)
        self.output = data_sig_gen.output
        self.data_gate_sig = data_sig_gen.gate_sig

        # Instruction TASK output
        task_sig_gen = PS_Sig_Gen(vocab.ps_task, 'PS TASK',
                                  cleanup_threshold=cfg.instr_ps_threshold)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], task_sig_gen.input,
            transform=instr_voc.parse('~TASK').get_convolution_matrix(),
            synapse=0.01)
        self.task_output = task_sig_gen.output
        self.task_gate_sig = task_sig_gen.gate_sig

        # Instruction STATE output
        state_sig_gen = PS_Sig_Gen(vocab.ps_state, 'PS STATE',
                                   cleanup_threshold=cfg.instr_ps_threshold)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], state_sig_gen.input,
            transform=instr_voc.parse('~STATE').get_convolution_matrix(),
            synapse=0.01)
        self.state_output = state_sig_gen.output
        self.state_gate_sig = state_sig_gen.gate_sig

        # Instruction DEC output
        dec_sig_gen = PS_Sig_Gen(vocab.ps_dec, 'PS DEC',
                                 cleanup_threshold=cfg.instr_ps_threshold)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], dec_sig_gen.input,
            transform=instr_voc.parse('~DEC').get_convolution_matrix(),
            synapse=0.01)
        self.dec_output = dec_sig_gen.output
        self.dec_gate_sig = dec_sig_gen.gate_sig

        # ------------- INSTRUCTION GATE OUTPUT ENABLE SIGNAL -----------------
        self.enable_in_sp = nengo.Node(size_in=vocab.instr.dimensions)

        self.gate_disable = cfg.make_thresh_ens_net()
        nengo.Connection(self.enable_in_sp, self.gate_disable.input,
                         transform=-vocab.instr.parse('ENABLE').v[:, None].T)
        nengo.Connection(bias_node, self.gate_disable.input)

        # Disable the outputs of the task, state, dec AM's when not ENABLEd
        nengo.Connection(self.gate_disable.output, task_sig_gen.inhibit)
        nengo.Connection(self.gate_disable.output, state_sig_gen.inhibit)
        nengo.Connection(self.gate_disable.output, dec_sig_gen.inhibit)

        # Connect gate disable signal to gate signal generators
        nengo.Connection(self.gate_disable.output, data_sig_gen.gate_sig_in,
                         transform=-80)
        nengo.Connection(self.gate_disable.output, task_sig_gen.gate_sig_in,
                         transform=-80)
        nengo.Connection(self.gate_disable.output, state_sig_gen.gate_sig_in,
                         transform=-80)
        nengo.Connection(self.gate_disable.output, dec_sig_gen.gate_sig_in,
                         transform=-80)

        # ------------- POS AM UTILITY OUTPUT (TO BG) ------------------------
        self.pos_util_output = nengo.Node(size_in=vocab.ps_task.dimensions)
        pos_util_matrix = np.array([vocab.ps_task.parse('INSTR').v] *
                                   len(vocab.pos.keys))
        nengo.Connection(pos_am.elem_utilities, self.pos_util_output,
                         transform=pos_util_matrix.T)

        # ------------------- MODULE INPUTS AND OUTPUTS -----------------------
        self.inputs = dict(en=(self.enable_in_sp, vocab.instr),
                           util=(self.pos_util_output, vocab.ps_task))
        self.outputs = dict(data=(self.output, vocab.main),
                            task=(self.task_output, vocab.ps_task),
                            state=(self.state_output, vocab.ps_state),
                            dec=(self.dec_output, vocab.ps_dec))

        # ## DEBUG ## #
        self.pos_am = pos_am
        self.instr_pos_cconv = instr_pos_cconv

        self.no_pos_chosen = no_pos_chosen

        def norm_func(t, x):
            return np.linalg.norm(x)

        self.norm_node1 = nengo.Node(size_in=vocab.sp_dim, output=norm_func)
        self.norm_node2 = nengo.Node(size_in=vocab.sp_dim, output=norm_func)
        nengo.Connection(instr_pos_cconv.A, self.norm_node1)
        nengo.Connection(instr_pos_cconv.B, self.norm_node2)

        self.task_node = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], self.task_node,
            transform=instr_voc.parse('~TASK').get_convolution_matrix(),
            synapse=0.01)

        self.state_node = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], self.state_node,
            transform=instr_voc.parse('~STATE').get_convolution_matrix(),
            synapse=0.01)

        self.dec_node = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(
            instr_cons_cconv.output[inv_conT], self.dec_node,
            transform=instr_voc.parse('~DEC').get_convolution_matrix(),
            synapse=0.01)

        self.pos_instr = instr_cons_cconv.B
        # self.pos_given = pos_given
        # ## DEBUG ## #

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            nengo.Connection(parent_net.vis.vis_main_mem.output,
                             self.vis_input, synapse=0.01)

            # ####### POS INC NETWORK #######
            pos_inc_gate_sp_vecs = vocab.main.parse('V').v
            pos_inc_rst_sp_vecs = vocab.main.parse('P+A+M').v

            nengo.Connection(parent_net.vis.output, self.pos_inc.gate,
                             transform=[pos_inc_gate_sp_vecs])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.pos_inc.gate, transform=-1.5)

            nengo.Connection(parent_net.vis.output, self.pos_inc.reset,
                             transform=[pos_inc_rst_sp_vecs])

            nengo.Connection(parent_net.vis.vis_main_mem.output,
                             self.pos_inc.input)
        else:
            warn("InstructionProcessingSystem Module - Cannot connect from " +
                 "'vis'")

        # Set up connections from vision module
        if hasattr(parent_net, 'instr_stim'):
            nengo.Connection(parent_net.instr_stim.output, self.instr_input)
        else:
            warn("InstructionProcessingSystem Module - Cannot connect from " +
                 "'instr_stim'")

        if hasattr(parent_net, 'ps'):
            nengo.Connection(parent_net.ps.task, self.task_input,
                             synapse=0.01)
            nengo.Connection(parent_net.ps.state, self.state_input,
                             synapse=0.01)
            nengo.Connection(parent_net.ps.dec, self.dec_input,
                             synapse=0.01)

            # ###### POS INC INIT STATE ######
            pos_inc_init_state_sp_vecs = vocab.ps_state.parse('INSTRP').v
            nengo.Connection(parent_net.ps.state, self.pos_inc_init.input,
                             transform=[pos_inc_init_state_sp_vecs])
        else:
            warn("InstructionProcessingSystem Module - Cannot connect from " +
                 "'ps'")
