from warnings import warn

import nengo
from nengo import spa
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab

from .spaun_module import SpaunModule, SpaunMPHub


class ProductionSystem(SpaunModule):
    def __init__(self, label="Production Sys", seed=None,
                 add_to_container=None):

        module_id_str = "ps"
        module_ind_num = 11

        super(ProductionSystem, self).__init__(
            module_id_str, module_ind_num, label, seed, add_to_container
        )

    @with_self
    def init_module(self):
        super().init_module()

        # Memory block to hold task information
        if cfg.ps_use_am_mb:
            self.task_mb = \
                cfg.make_mem_block(vocab=vocab.ps_task,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.5, wta_output=False,
                                   reset_key="X")

            self.state_mb = \
                cfg.make_mem_block(vocab=vocab.ps_state,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.3, wta_output=True,
                                   wta_inhibit_scale=3, reset_key="TRANS0")

            self.dec_mb = \
                cfg.make_mem_block(vocab=vocab.ps_dec,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.3, wta_output=True,
                                   wta_inhibit_scale=3, reset_key="FWD")
        else:
            self.task_mb = \
                cfg.make_mem_block(vocab=vocab.ps_task,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key="X")

            self.state_mb = \
                cfg.make_mem_block(vocab=vocab.ps_state,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key="TRANS0")

            self.dec_mb = \
                cfg.make_mem_block(vocab=vocab.ps_dec,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key="FWD")

        # ------ Associative memory for non-mb actions ------
        self.action_in = nengo.Node(size_in=vocab.ps_action.dimensions)
        self.action_am = \
            cfg.make_assoc_mem(input_vectors=vocab.ps_action.vectors,
                               threshold=cfg.ps_action_am_threshold)  # ,
                               # wta_inhibit_scale=None)  # noqa
        nengo.Connection(self.action_in, self.action_am.input, synapse=0.01)

        # ----- Task initialization ("X") phase -----
        # Create threshold ensemble to handle initialization of tasks
        # - task mb gate signal is set high when in init state.
        # - state mb gate signal is set high when in init state.
        # - dec mb gate signal is set high when in init state.
        self.task_init = cfg.make_thresh_ens_net(0.4)

        nengo.Connection(self.task_init.output, self.task_mb.gate)
        nengo.Connection(self.task_init.output, self.state_mb.gate)
        nengo.Connection(self.task_init.output, self.dec_mb.gate)

        ps_task_init_task_sp_vecs = vocab.main.parse("X").v
        nengo.Connection(self.task_mb.output, self.task_init.input,
                         transform=[ps_task_init_task_sp_vecs],
                         synapse=0.01)

        # ------ Gate signal from decoding system ------
        # Decoding gate signal for changing state and dec mb's in the DEC phase
        self.dec_sys_gate_sig = cfg.make_thresh_ens_net(0.5)
        nengo.Connection(self.dec_sys_gate_sig.output, self.state_mb.gate,
                         transform=5)
        nengo.Connection(self.dec_sys_gate_sig.output, self.dec_mb.gate,
                         transform=5)

        # Ignore the decoding system gate signal when task is INSTR
        no_dec_sys_gate_sig_sp_vecs = vocab.ps_task.parse("INSTR").v
        nengo.Connection(self.task_mb.output, self.dec_sys_gate_sig.input,
                         transform=[-2.0 * no_dec_sys_gate_sig_sp_vecs],
                         synapse=0.01)

    @with_self
    def setup_inputs_and_outputs(self):
        # ------ Define inputs and outputs ------
        self.expose_input("task", self.task_mb.input)
        self.expose_input("state", self.state_mb.input)
        self.expose_input("dec", self.dec_mb.input)
        self.expose_input("action", self.action_in)
        self.expose_output("task", self.task_mb.output)
        self.expose_output("state", self.state_mb.output)
        self.expose_output("dec", self.dec_mb.output)
        self.expose_output("action", self.action_am.output)

        # ------ Expose inputs for external connections ------
        if cfg.has_vis:
            self.add_module_input("vis", "main", vocab.sp_dim)
            self.add_module_input("vis", "neg_attn", 1)

            # ###### Task MB ########
            task_mb_gate_sp_vecs = vocab.main.parse("QM+M+V").v
            task_mb_rst_sp_vecs = vocab.main.parse("A").v

            nengo.Connection(self.get_inp("vis_main"), self.task_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        task_mb_gate_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"),
                             self.task_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(self.get_inp("vis_main"), self.task_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        task_mb_rst_sp_vecs])

            # ###### State MB ########
            state_mb_gate_sp_vecs = vocab.main.parse("CLOSE+M+V+K+P+QM").v
            state_mb_rst_sp_vecs = vocab.main.parse("0").v

            nengo.Connection(self.get_inp("vis_main"), self.state_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        state_mb_gate_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"),
                             self.state_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(self.get_inp("vis_main"), self.state_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        state_mb_rst_sp_vecs])

            # ###### Dec MB ########
            dec_mb_gate_sp_vecs = vocab.main.parse("M+V+F+R+QM").v
            dec_mb_rst_sp_vecs = vocab.main.parse("0").v

            nengo.Connection(self.get_inp("vis_main"), self.dec_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        dec_mb_gate_sp_vecs])
            nengo.Connection(self.get_inp("vis_neg_attn"),
                             self.dec_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(self.get_inp("vis_main"), self.dec_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        dec_mb_rst_sp_vecs])

        if cfg.has_trfm:
            self.add_module_input("trfm", "compare_gate", 1)

            nengo.Connection(self.get_inp("trfm_compare_gate"),
                             self.state_mb.gate, transform=4)
            nengo.Connection(self.get_inp("trfm_compare_gate"),
                             self.dec_mb.gate, transform=4)

        if cfg.has_dec:
            self.add_module_input("dec" ,"pos_gate", self.dec_sys_gate_sig.input)

        if cfg.has_instr:
            self.add_module_input("instr", "task_gate", 1)
            self.add_module_input("instr", "state_gate", 1)
            self.add_module_input("instr", "dec_gate", 1)
            self.add_module_input("instr", "pos_inc_init", 1)

            nengo.Connection(self.get_inp("instr_task_gate"),
                             self.task_mb.gate, transform=5)
            nengo.Connection(self.get_inp("instr_state_gate"),
                             self.state_mb.gate, transform=5)
            nengo.Connection(self.get_inp("instr_dec_gate"),
                             self.dec_mb.gate, transform=5)

            # ###### POS INC INIT STATE ######
            nengo.Connection(self.get_inp("instr_pos_inc_init"),
                             self.state_mb.gate, transform=2)

    def setup_spa_inputs_and_outputs(self):
        # ------ Define SPA module input and outputs ------
        self.inputs = dict(task=(self.get_inp("task"), vocab.ps_task),
                           state=(self.get_inp("state"), vocab.ps_state),
                           dec=(self.get_inp("dec"), vocab.ps_dec),
                           action=(self.get_inp("action"), vocab.ps_action))
        self.outputs = dict(task=(self.get_out("task"), vocab.ps_task),
                            state=(self.get_out("state"), vocab.ps_state),
                            dec=(self.get_out("dec"), vocab.ps_dec),
                            action=(self.get_out("action"), vocab.ps_action))

    def get_multi_process_hub(self):
        return ProductionSystemMPHub(self)


class ProductionSystemMPHub(SpaunMPHub, ProductionSystem):
    def __init__(self, parent_module):
        SpaunMPHub.__init__(self, parent_module)
