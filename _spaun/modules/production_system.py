from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import ps_task_vocab, ps_state_vocab, ps_dec_vocab
from ..vocabs import ps_task_mb_gate_sp_inds, ps_task_mb_rst_sp_inds
from ..vocabs import ps_state_mb_gate_sp_inds, ps_state_mb_rst_sp_inds
from ..vocabs import ps_dec_mb_gate_sp_inds, ps_dec_mb_rst_sp_inds


class ProductionSystem(Module):
    def __init__(self, label="Prod Sys", seed=None, add_to_container=None):
        super(ProductionSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        # Memory block to hold task information
        self.ps_task_mb = cfg.make_mem_block(vocab=ps_task_vocab,
                                             cleanup_mode=1,
                                             reset_key='X')

        self.ps_state_mb = cfg.make_mem_block(vocab=ps_state_vocab,
                                              cleanup_mode=1,
                                              reset_key='INIT')

        self.ps_dec_mb = cfg.make_mem_block(vocab=ps_dec_vocab,
                                            cleanup_mode=1,
                                            reset_key='FWD')

        # Define inputs and outputs
        self.task = self.ps_task_mb.output
        self.state = self.ps_state_mb.output
        self.dec = self.ps_dec_mb.output

        # Define module input and outputs
        self.inputs = dict(task=(self.ps_task_mb.input, ps_task_vocab),
                           state=(self.ps_state_mb.input, ps_state_vocab),
                           dec=(self.ps_dec_mb.input, ps_dec_vocab))
        self.outputs = dict(task=(self.ps_task_mb.output, ps_task_vocab),
                            state=(self.ps_state_mb.output, ps_state_vocab),
                            dec=(self.ps_dec_mb.output, ps_dec_vocab))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            # ###### Task MB ########
            nengo.Connection(
                parent_net.vis.am_utilities[ps_task_mb_gate_sp_inds],
                self.ps_task_mb.gate,
                transform=[[cfg.mb_gate_scale] * len(ps_task_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.ps_task_mb.gate, transform=-2, synapse=0.01)

            nengo.Connection(
                parent_net.vis.am_utilities[ps_task_mb_rst_sp_inds],
                self.ps_task_mb.reset,
                transform=[[cfg.mb_gate_scale] * len(ps_task_mb_rst_sp_inds)])

            # ###### State MB ########
            nengo.Connection(
                parent_net.vis.am_utilities[ps_state_mb_gate_sp_inds],
                self.ps_state_mb.gate,
                transform=[[cfg.mb_gate_scale] *
                           len(ps_state_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.ps_state_mb.gate, transform=-2, synapse=0.01)

            nengo.Connection(
                parent_net.vis.am_utilities[ps_state_mb_rst_sp_inds],
                self.ps_state_mb.reset,
                transform=[[cfg.mb_gate_scale] * len(ps_state_mb_rst_sp_inds)])

            # ###### Dec MB ########
            nengo.Connection(
                parent_net.vis.am_utilities[ps_dec_mb_gate_sp_inds],
                self.ps_dec_mb.gate,
                transform=[[cfg.mb_gate_scale] * len(ps_dec_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.ps_dec_mb.gate, transform=-2, synapse=0.01)

            nengo.Connection(
                parent_net.vis.am_utilities[ps_dec_mb_rst_sp_inds],
                self.ps_dec_mb.reset,
                transform=[[cfg.mb_gate_scale] * len(ps_dec_mb_rst_sp_inds)])
        else:
            warn("ProductionSystem Module - Cannot connect from 'vis'")
