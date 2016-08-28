from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab


class ProductionSystem(Module):
    def __init__(self, label="Production Sys", seed=None,
                 add_to_container=None):
        super(ProductionSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        # Memory block to hold task information
        if cfg.ps_use_am_mb:
            self.task_mb = \
                cfg.make_mem_block(vocab=vocab.ps_task,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.5, wta_output=False,
                                   reset_key='X')

            self.state_mb = \
                cfg.make_mem_block(vocab=vocab.ps_state,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.3, wta_output=True,
                                   reset_key='TRANS0')

            self.dec_mb = \
                cfg.make_mem_block(vocab=vocab.ps_dec,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   cleanup_mode=1, fdbk_transform=1.05,
                                   threshold=0.3, wta_output=True,
                                   reset_key='FWD')
        else:
            self.task_mb = \
                cfg.make_mem_block(vocab=vocab.ps_task,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key='X')

            self.state_mb = \
                cfg.make_mem_block(vocab=vocab.ps_state,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key='TRANS0')

            self.dec_mb = \
                cfg.make_mem_block(vocab=vocab.ps_dec,
                                   input_transform=cfg.ps_mb_gain_scale,
                                   fdbk_transform=1.005, reset_key='FWD')

        # ------ Associative memory for non-mb actions ------
        self.action_in = nengo.Node(size_in=vocab.ps_action.dimensions)
        self.action_am = \
            cfg.make_assoc_mem(input_vectors=vocab.ps_action.vectors,
                               threshold=cfg.ps_action_am_threshold)  # ,
                               # wta_inhibit_scale=None)  # noqa
        nengo.Connection(self.action_in, self.action_am.input, synapse=0.01)

        # ------ Define inputs and outputs ------
        self.task = self.task_mb.output
        self.state = self.state_mb.output
        self.dec = self.dec_mb.output
        self.action = self.action_am.output

        # ----- Task initialization ('X') phase -----
        # Create threshold ensemble to handle initialization of tasks
        # - task mb gate signal is set high when in init state.
        # - state mb gate signal is set high when in init state.
        # - dec mb gate signal is set high when in init state.
        self.task_init = cfg.make_thresh_ens_net(0.4)

        nengo.Connection(self.task_init.output, self.task_mb.gate)
        nengo.Connection(self.task_init.output, self.state_mb.gate)
        nengo.Connection(self.task_init.output, self.dec_mb.gate)

        ps_task_init_task_sp_vecs = vocab.main.parse('X').v
        nengo.Connection(self.task, self.task_init.input,
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
        no_dec_sys_gate_sig_sp_vecs = vocab.ps_task.parse('INSTR').v
        nengo.Connection(self.task, self.dec_sys_gate_sig.input,
                         transform=[-2.0 * no_dec_sys_gate_sig_sp_vecs],
                         synapse=0.01)

        # ------ Define module input and outputs ------
        self.inputs = dict(task=(self.task_mb.input, vocab.ps_task),
                           state=(self.state_mb.input, vocab.ps_state),
                           dec=(self.dec_mb.input, vocab.ps_dec),
                           action=(self.action_in, vocab.ps_action))
        self.outputs = dict(task=(self.task_mb.output, vocab.ps_task),
                            state=(self.state_mb.output, vocab.ps_state),
                            dec=(self.dec_mb.output, vocab.ps_dec),
                            action=(self.action_am.output, vocab.ps_action))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            # ###### Task MB ########
            task_mb_gate_sp_vecs = vocab.main.parse('QM+M').v
            task_mb_rst_sp_vecs = vocab.main.parse('A').v

            nengo.Connection(parent_net.vis.output, self.task_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        task_mb_gate_sp_vecs])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.task_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(parent_net.vis.output, self.task_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        task_mb_rst_sp_vecs])

            # ###### State MB ########
            state_mb_gate_sp_vecs = vocab.main.parse('CLOSE+K+P+QM').v
            state_mb_rst_sp_vecs = vocab.main.parse('0').v

            nengo.Connection(parent_net.vis.output, self.state_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        state_mb_gate_sp_vecs])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.state_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(parent_net.vis.output, self.state_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        state_mb_rst_sp_vecs])

            # ###### Dec MB ########
            dec_mb_gate_sp_vecs = vocab.main.parse('F+R+QM').v
            dec_mb_rst_sp_vecs = vocab.main.parse('0').v

            nengo.Connection(parent_net.vis.output, self.dec_mb.gate,
                             transform=[cfg.ps_mb_gate_scale *
                                        dec_mb_gate_sp_vecs])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.dec_mb.gate, transform=-1.5, synapse=0.005)

            nengo.Connection(parent_net.vis.output, self.dec_mb.reset,
                             transform=[cfg.ps_mb_gate_scale *
                                        dec_mb_rst_sp_vecs])
        else:
            warn("ProductionSystem Module - Cannot connect from 'vis'")

        # Set up connections from dec module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.pos_mb_gate_sig.output,
                             self.dec_sys_gate_sig.input)
        else:
            warn("ProductionSystem Module - Could not connect from 'dec'")

        # Set up connections from instr module
        if hasattr(parent_net, 'instr'):
            nengo.Connection(parent_net.instr.task_gate_sig,
                             self.task_mb.gate, transform=5)
            nengo.Connection(parent_net.instr.state_gate_sig,
                             self.state_mb.gate, transform=5)
            nengo.Connection(parent_net.instr.dec_gate_sig,
                             self.dec_mb.gate, transform=5)

            # ###### POS INC INIT STATE ######
            nengo.Connection(parent_net.instr.pos_inc_init.output,
                             self.state_mb.gate, transform=2)
        else:
            warn("ProductionSystem Module - Could not connect from 'instr'")
