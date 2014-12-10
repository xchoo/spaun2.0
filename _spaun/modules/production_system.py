from warnings import warn

import nengo
from nengo.spa.module import Module

from ..config import cfg
from ..vocabs import task_mb_gate_sp_inds, task_mb_rst_sp_inds
from ..vocabs import task_vocab


class ProductionSystem(Module):
    def __init__(self):
        super(ProductionSystem, self).__init__()

        # Memory block to hold task information
        self.task_mb = cfg.make_mem_block(cleanup_vecs=task_vocab.vectors,
                                          reset_vec=task_vocab['A'].v)

        # Define inputs and outputs
        self.task = self.task_mb.output

        # Define module input and outputs
        self.inputs = dict(task=(self.task_mb.input, task_vocab))
        self.outputs = dict(task=(self.task_mb.output, task_vocab))

    def setup_connections(self, parent_net):
        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            nengo.Connection(parent_net.vis.am_utilities[task_mb_gate_sp_inds],
                             self.task_mb.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(task_mb_gate_sp_inds)])
            nengo.Connection(parent_net.vis.neg_attention,
                             self.task_mb.gate, transform=-2, synapse=0.01)

            nengo.Connection(parent_net.vis.am_utilities[task_mb_rst_sp_inds],
                             self.task_mb.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(task_mb_rst_sp_inds)])
        else:
            warn("ProductionSystem Module - Cannot connect from 'vis'")
