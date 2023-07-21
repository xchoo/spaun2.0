from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import Selector
from ..configurator import cfg
from ..vocabulator import vocab

from .spaun_module import SpaunModule, SpaunMPHub
from .memory import WM_Generic_Network, WM_Averaging_Network


class WorkingMemory(SpaunModule):
    def __init__(self, label="Working Memory", seed=None,
                 add_to_container=None):

        module_id_str = "mem"
        module_ind_num = 14

        super(WorkingMemory, self).__init__(
            module_id_str, module_ind_num, label, seed, add_to_container
        )

    @with_self
    def init_module(self):
        super().init_module()

        # --------------------------- Bias nodes ---------------------------- #
        self.bias_node = nengo.Node(1, label="Bias")

        # -------------------------- Gate signals --------------------------- #
        # Common gate signal
        self.wm_gate = cfg.make_thresh_ens_net(0.25)
        nengo.Connection(self.bias_node, self.wm_gate.input, transform=1.2)

        # Memory input selector
        self.select_in = cfg.make_selector(
            2, default_sel=0, make_ens_func=cfg.make_spa_ens_array)

        # Gate signal selector
        self.select_gate = Selector(cfg.n_neurons_ens, dimensions=1,
                                    num_items=2, make_ens_func=nengo.Ensemble,
                                    gate_gain=10, default_sel=0,
                                    threshold_sel_in=True)

        # "ADD" transform for WM
        sp_add_matrix = (vocab.add_sp.get_convolution_matrix() *
                         (0.25 / cfg.mb_rehearsalbuf_input_scale +
                          0.25 / (cfg.mb_decaybuf_input_scale - 0.15)))

        self.num0_bias_node = nengo.Node(vocab.main.parse("POS1*ZER").v,
                                         label="POS1*ZER")

        self.gate_sig_bias = cfg.make_thresh_ens_net(label="Gate Sig Bias")
        # Bias the -1.5 neg_atn during decoding phase (when there is no input)
        # nengo.Connection(self.gate_sig_bias.output, self.select_gate.input0)

        self.cnt_gate_sig = cfg.make_thresh_ens_net(0.5, label="Cnt Gate Sig")
        nengo.Connection(self.cnt_gate_sig.output, self.select_gate.input0,
                         transform=cfg.mb_gate_scale)

        # ------------------------- Memory block 1 ------------------------- #
        self.mb1_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB1")
        nengo.Connection(self.select_in.output, self.mb1_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb1_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb1_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb1_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb1_net.gate,
        #                  transform=1.0)

        self.mb1 = self.mb1_net.output

        # ------------------------- Memory block 2 ------------------------- #
        self.mb2_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB2")
        nengo.Connection(self.select_in.output, self.mb2_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb2_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb2_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb2_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb2_net.gate,
        #                  transform=1.0)

        self.mb2 = self.mb2_net.output

        # ------------------------- Memory block 3 ------------------------- #
        self.mb3_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB3")
        nengo.Connection(self.select_in.output, self.mb3_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb3_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb3_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb3_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb3_net.gate,
        #                  transform=1.0)

        self.mb3 = self.mb3_net.output

        # -------------------- Memory block Ave (MBAve) -------------------- #
        self.mbave_net = WM_Averaging_Network(vocab.main)
        self.mbave = self.mbave_net.output

        # Define network inputs and outputs
        self.input = self.select_in.input0
        self.data_input = self.select_in.input1

        self.gate_in = self.wm_gate.input
        nengo.Connection(self.wm_gate.output, self.select_gate.input0)

        self.data_gate_in = self.select_gate.input1

        # ################ DEBUG ################
        self.gate_in_neg_att_dbg = nengo.Node(size_in=1)
        self.gate_in_vis_dbg = nengo.Node(size_in=1)
        self.gate_in_2 = nengo.Node(size_in=1)
        nengo.Connection(self.gate_sig_bias.output, self.gate_in_2)
        nengo.Connection(self.cnt_gate_sig.output, self.gate_in_2,
                         transform=cfg.mb_gate_scale)
        self.gate_sel_none = self.select_gate.sel_none
        self.gate_sel_node0 = self.select_gate.sel_nodes[0]
        self.gate_sel_node1 = self.select_gate.sel_nodes[1]
        self.gate_sel_node1_in = nengo.Node(size_in=1)
        # ################ DEBUG ################

    @with_self
    def setup_inputs_and_outputs(self):
        # ------ Define inputs and outputs ------
        self.expose_output("mb1", self.mb1)
        self.expose_output("mb2", self.mb2)
        self.expose_output("mb3", self.mb3)
        self.expose_output("mbave", self.mbave)

        # ------ Expose inputs for external connections ------
        # Set up connections from vision module
        if cfg.has_vis:
            self.add_module_input("vis", "main", vocab.sp_dim)
            self.add_module_input("vis", "neg_attn", 1)

            # ###### Common Gate Input Signal ######
            item_mb_no_gate_sp_vecs = \
                vocab.main.parse("+".join(vocab.ps_task_vis_sp_strs +
                                          vocab.misc_vis_sp_strs)).v
            item_mb_rst_sp_vecs = vocab.main.parse("A+OPEN").v

            nengo.Connection(self.get_inp("vis_main"), self.gate_in,
                             transform=[-cfg.mb_gate_scale *
                                        item_mb_no_gate_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"), self.gate_in,
                             transform=-cfg.mb_neg_attn_scale * 2.0,
                             synapse=0.02)

            # ###### MB1 ########
            nengo.Connection(self.get_inp("vis_main"), self.mb1_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"), self.mb1_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MB2 ########
            nengo.Connection(self.get_inp("vis_main"), self.mb2_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"), self.mb2_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MB3 ########
            nengo.Connection(self.get_inp("vis_main"), self.mb3_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"), self.mb3_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MBAve ########
            ave_mb_gate_sp_vecs = vocab.main.parse("CLOSE").v
            ave_mb_rst_sp_vecs = vocab.main.parse("A").v

            nengo.Connection(self.get_inp("vis_main"), self.mbave_net.gate,
                             transform=[cfg.mb_gate_scale *
                                        ave_mb_gate_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"),
                             self.mbave_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            nengo.Connection(self.get_inp("vis_main"), self.mbave_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        ave_mb_rst_sp_vecs])

        # Set up connections from production system module
        if cfg.has_ps:
            self.add_module_input("ps", "task", vocab.sp_dim)
            self.add_module_input("ps", "state", vocab.sp_dim)
            self.add_module_input("ps", "dec", vocab.sp_dim)

            # ###### INPUT SELECTOR #######
            instr_task_sp_vecs = vocab.main.parse("INSTR").v
            nengo.Connection(self.get_inp("ps_task"), self.select_in.sel1,
                             transform=[instr_task_sp_vecs])
            nengo.Connection(self.get_inp("ps_task"), self.select_gate.sel1,
                             transform=[instr_task_sp_vecs])

            # ################ DEBUG ################
            # nengo.Connection(self.get_inp("ps_task"), self.gate_sel_node1_in,
            #                  transform=[instr_task_sp_vecs])
            # ################ DEBUG ################

            # Note: Thresholded ensembles used because addition of semantic
            #       pointers can cause negative values (which we don't
            #       want). Two thresholded ensembles are used to too
            #       increase detection threshold for semantic pointers.
            #       (i.e. hopefully, the addition of all of the semantic
            #        pointers aren't so negative as to override the
            #        positive semantic pointer value)
            # --> Maximum of 5 semantic pointer additions per thresholded
            #     ensemble

            # ###### MB1 ########
            mb1_no_gate_thresh_ens1 = cfg.make_thresh_ens_net()
            mb1_no_gate_sp_vecs1 = \
                vocab.main.parse("QAP+QAK+TRANS1+TRANS2+CNT0").v
            nengo.Connection(self.get_inp("ps_state"), mb1_no_gate_thresh_ens1.input,
                             transform=[mb1_no_gate_sp_vecs1])
            nengo.Connection(mb1_no_gate_thresh_ens1.output,
                             self.mb1_net.gate,
                             transform=cfg.mb_neg_gate_scale)

            mb1_no_gate_thresh_ens2 = cfg.make_thresh_ens_net()
            mb1_no_gate_sp_vecs2 = \
                vocab.main.parse("X+L").v
            nengo.Connection(self.get_inp("ps_task"), mb1_no_gate_thresh_ens2.input,
                             transform=[mb1_no_gate_sp_vecs2])
            nengo.Connection(mb1_no_gate_thresh_ens2.output,
                             self.mb1_net.gate,
                             transform=cfg.mb_neg_gate_scale)

            mb1_no_reset_thresh_ens = cfg.make_thresh_ens_net()
            mb1_no_reset_sp_vecs = \
                vocab.main.parse("QAP+QAK+TRANS1+CNT0+CNT1").v
            nengo.Connection(self.get_inp("ps_state"), mb1_no_reset_thresh_ens.input,
                             transform=[mb1_no_reset_sp_vecs])
            nengo.Connection(mb1_no_reset_thresh_ens.output,
                             self.mb1_net.reset,
                             transform=[cfg.mb_neg_gate_scale])

            # ################ DEBUG ################
            self.mb1_no_gate_in = mb1_no_gate_thresh_ens1.input
            self.mb1_no_gate_out = mb1_no_gate_thresh_ens1.output
            # ################ DEBUG ################

            mb1_sel_1_sp_vecs = vocab.main.parse("CNT1").v
                # Use *ONE connection in the CNT1 state  # noqa
            nengo.Connection(self.get_inp("ps_state"), self.mb1_net.sel1,
                             transform=[mb1_sel_1_sp_vecs])
            nengo.Connection(self.get_inp("ps_state"), self.mb1_net.fdbk_gate,
                             transform=[mb1_sel_1_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(self.get_inp("ps_task"), self.mb1_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MB2 ########
            mb2_no_gate_thresh_ens = cfg.make_thresh_ens_net()
            mb2_no_gate_sp_vecs = \
                vocab.main.parse("X+TRANS0+TRANS2+CNT1+L").v
            nengo.Connection(self.get_inp("ps_state"), mb2_no_gate_thresh_ens.input,
                             transform=[mb2_no_gate_sp_vecs])
            nengo.Connection(self.get_inp("ps_task"), mb2_no_gate_thresh_ens.input,
                             transform=[mb2_no_gate_sp_vecs])
            nengo.Connection(mb2_no_gate_thresh_ens.output,
                             self.mb2_net.gate,
                             transform=cfg.mb_neg_gate_scale)

            mb2_no_reset_thresh_ens = cfg.make_thresh_ens_net()
            mb2_no_reset_sp_vecs = \
                vocab.main.parse("TRANS2+CNT1+TRANSC").v
            # mb2_no_reset_sp_vecs = \
            #     vocab.main.parse("QAP+QAK+TRANS2+CNT1+TRANSC").v
            # Why is there a no reset for the QAP and QAK states??
            #    vocab.main.parse("QAP+QAK+TRANS1+TRANS2+CNT1+TRANSC").v
            # Why is there a no reset for the TRANS1 state???
            nengo.Connection(self.get_inp("ps_state"), mb2_no_reset_thresh_ens.input,
                             transform=[mb2_no_reset_sp_vecs])
            nengo.Connection(mb2_no_reset_thresh_ens.output,
                             self.mb2_net.reset,
                             transform=[cfg.mb_neg_gate_scale])

            mb2_sel_1_sp_vecs = vocab.main.parse("0").v
            # TODO: Make configurable? Use *ONE connection in the none
            nengo.Connection(self.get_inp("ps_state"), self.mb2_net.sel1,
                             transform=[mb2_sel_1_sp_vecs])
            nengo.Connection(self.get_inp("ps_state"), self.mb2_net.fdbk_gate,
                             transform=[mb2_sel_1_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(self.get_inp("ps_task"), self.mb2_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MB3 ########
            mb3_no_gate_thresh_ens = cfg.make_thresh_ens_net()
            mb3_no_gate_sp_vecs = \
                vocab.main.parse("X+QAP+QAK+TRANS0+TRANS1+L").v
            nengo.Connection(self.get_inp("ps_state"), mb3_no_gate_thresh_ens.input,
                             transform=[mb3_no_gate_sp_vecs])
            nengo.Connection(self.get_inp("ps_task"), mb3_no_gate_thresh_ens.input,
                             transform=[mb3_no_gate_sp_vecs])
            nengo.Connection(mb3_no_gate_thresh_ens.output,
                             self.mb3_net.gate,
                             transform=cfg.mb_neg_gate_scale)

            mb3_no_reset_thresh_ens = cfg.make_thresh_ens_net()
            mb3_no_reset_sp_vecs = vocab.main.parse("CNT1+TRANSC").v
            nengo.Connection(self.get_inp("ps_state"), mb3_no_reset_thresh_ens.input,
                             transform=[mb3_no_reset_sp_vecs])
            nengo.Connection(mb3_no_reset_thresh_ens.output,
                             self.mb3_net.reset,
                             transform=[cfg.mb_neg_gate_scale])

            mb3_sel_1_sp_vecs = vocab.main.parse("CNT1").v
                # Use *ONE connection in the CNT1 state  # noqa
            nengo.Connection(self.get_inp("ps_state"), self.mb3_net.sel1,
                             transform=[mb3_sel_1_sp_vecs])
            nengo.Connection(self.get_inp("ps_state"), self.mb3_net.fdbk_gate,
                             transform=[mb3_sel_1_sp_vecs])

            mb3_sel_2_sp_vecs = vocab.main.parse("CNT0").v
                # Use POS1*ONE connection for CNT0 state  # noqa
            nengo.Connection(self.get_inp("ps_state"), self.mb3_net.sel2,
                             transform=[mb3_sel_2_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(self.get_inp("ps_task"), self.mb3_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MBAVe ########
            mbave_no_gate_thresh_ens = cfg.make_thresh_ens_net()
            mbave_no_gate_sp_vecs = \
                vocab.main.parse("X+QAP+QAK+TRANS0+L").v
            nengo.Connection(self.get_inp("ps_state"),
                             mbave_no_gate_thresh_ens.input,
                             transform=[mbave_no_gate_sp_vecs])
            nengo.Connection(self.get_inp("ps_task"), mbave_no_gate_thresh_ens.input,
                             transform=[mbave_no_gate_sp_vecs])
            nengo.Connection(mbave_no_gate_thresh_ens.output,
                             self.mbave_net.gate,
                             transform=cfg.mb_neg_gate_scale)

            mbave_do_reset_sp_vecs = vocab.main.parse("X").v
            nengo.Connection(self.get_inp("ps_task"), self.mbave_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        mbave_do_reset_sp_vecs])

            # ###### Gate Signal Bias ######
            gate_sig_bias_enable_sp_vecs = vocab.main.parse("CNT").v
                # Only enable gate signal bias for dec=CNT  # noqa
            nengo.Connection(self.get_inp("ps_dec"), self.gate_sig_bias.input,
                             transform=[gate_sig_bias_enable_sp_vecs],
                             synapse=0.01)

        # Set up connections from encoding module
        if cfg.has_enc:
            self.add_module_input("enc", "out", self.input)

        # Set up connections from transformation system module
        if cfg.has_trfm:
            self.add_module_input("trfm", "out", self.mbave_net.input)

        # Set up connections from instruction processing module
        if cfg.has_instr:
            self.add_module_input("instr", "out", self.data_input)
            self.add_module_input("instr", "data_gate", 1)

            nengo.Connection(self.get_inp("instr_data_gate"), self.data_gate_in,
                             transform=2.0, synapse=0.01)
            # Note: Need to filter the data_gate_sig signal a bit. Change
            #       detect mechanism is a little sensitive.

        # Set up connections from motor module (for counting task)
        if cfg.has_mtr:
            self.add_module_input("mtr", "ramp_reset_hold", 1)
            self.add_module_input("mtr", "ramp_init_hold", 1)

            nengo.Connection(self.get_inp("mtr_ramp_reset_hold"),
                             self.cnt_gate_sig.input, transform=2.0,
                             synapse=0.01)

            nengo.Connection(self.get_inp("mtr_ramp_init_hold"),
                             self.gate_in, transform=-2.0)

    def setup_spa_inputs_and_outputs(self):
        # ----- Set up module vocab inputs and outputs -----
        self.outputs = dict(mb1=(self.get_out("mb1"), vocab.enum),
                            mb2=(self.get_out("mb2"), vocab.enum),
                            mb3=(self.get_out("mb3"), vocab.enum),
                            mbave=(self.get_out("mbave"), vocab.enum))

    def get_multi_process_hub(self):
        return WorkingMemoryMPHub(self)


# class WorkingMemoryDummy(WorkingMemory):
#     def __init__(self):
#         super(WorkingMemoryDummy, self).__init__()
#         self.init_module()

#     @with_self
#     def init_module(self):
#         # Memory input node
#         self.mem_in = nengo.Node(size_in=vocab.sp_dim,
#                                  label="WM Module In Node")

#         self.gate_sig_bias = nengo.Node(size_in=1)

#         # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
#         self.mb1 = \
#             nengo.Node(output=vocab.main.parse("POS1*FOR+POS2*THR+POS3*FOR").v)
#         self.mb1_net.gate = nengo.Node(size_in=1, label="MB1 Gate Node")
#         self.mb1_net.reset = nengo.Node(size_in=1, label="MB1 Reset Node")

#         self.sel_mb1_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
#                                             ens_dimensions=vocab.sp_dim,
#                                             n_neurons=vocab.sp_dim)
#         nengo.Connection(self.mem_in, self.sel_mb1_in.input0, synapse=None)

#         # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
#         self.mb2 = nengo.Node(output=vocab.main.parse("POS1*THR").v)
#         self.mb2_gate = nengo.Node(size_in=1, label="MB2 Gate Node")
#         self.mb2_reset = nengo.Node(size_in=1, label="MB2 Reset Node")

#         self.sel_mb2_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
#                                             ens_dimensions=vocab.sp_dim,
#                                             n_neurons=vocab.sp_dim)
#         nengo.Connection(self.mem_in, self.sel_mb2_in.input0, synapse=None)

#         # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
#         self.mb3 = nengo.Node(output=vocab.main.parse("POS1*ONE").v)
#         self.mb3_gate = nengo.Node(size_in=1, label="MB3 Gate Node")
#         self.mb3_reset = nengo.Node(size_in=1, label="MB3 Reset Node")

#         self.sel_mb3_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
#                                             ens_dimensions=vocab.sp_dim,
#                                             n_neurons=vocab.sp_dim)
#         nengo.Connection(self.mem_in, self.sel_mb3_in.input0, synapse=None)

#         # Memory block Ave (MBAve)
#         self.mbave_in = nengo.Node(size_in=vocab.sp_dim)
#         self.mbave = nengo.Node(output=vocab.main.parse("~POS1").v)
#         self.mbave_gate = nengo.Node(size_in=1)
#         self.mbave_reset = nengo.Node(size_in=1)

#         # Define network inputs and outputs
#         # ## TODO: Fix this! (update to include selector and what not)
#         self.input = self.mem_in


class WorkingMemoryMPHub(SpaunMPHub, WorkingMemory):
    def __init__(self, parent_module):
        SpaunMPHub.__init__(self, parent_module)
