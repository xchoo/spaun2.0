import nengo
from nengo.spa.module import Module

from .._spa import MemoryBlock as MB

from ..config import cfg
from ..vocabs import task_mb_gate_sp_inds
from ..vocabs import task_vocab


class ProductionSystem(Module):
    def __init__(self):
        super(ProductionSystem, self).__init__()

        # Memory block to hold task information
        self.task_mb = MB(cfg.n_neurons_mb, cfg.sp_dim, gate_mode=2)

        # Define inputs and outputs
        self.task = self.task_mb.output

        # Define module input and outputs
        self.inputs = dict(task=(self.task_mb.input, task_vocab))
        self.outputs = dict(task=(self.task_mb.output, task_vocab))

    def connect_from_vision(self, vision_module):
        nengo.Connection(vision_module.am_utilities[task_mb_gate_sp_inds],
                         self.task_mb.gate,
                         transform=[[1] * len(task_mb_gate_sp_inds)])
        nengo.Connection(vision_module.neg_attention,
                         self.task_mb.gate, transform=-1, synapse=0.01)
