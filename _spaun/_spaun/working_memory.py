import nengo
from nengo.spa.module import Module

from ..config import cfg
from .._spa import MemoryBlock as MB
from .._vocab.vocabs import item_mb_gate_sp_inds
from .._vocab.vocabs import item_mb_rst_sp_inds


class WorkingMemory(Module):
    def __init__(self):
        super(WorkingMemory, self).__init__()

        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim)

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1 = nengo.Node(size_in=cfg.sp_dim)
        self.mb1a = MB(cfg.n_neurons_mb, cfg.sp_dim, gate_mode=2, label="MB1A")
        self.mb1b = MB(cfg.n_neurons_mb, cfg.sp_dim, gate_mode=2, label="MB1B",
                       fdbk_scale=cfg.mb_decay_val)
        nengo.Connection(self.mem_in, self.mb1a.input)
        nengo.Connection(self.mem_in, self.mb1b.input)
        nengo.Connection(self.mb1a.output, self.mb1a.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(self.mb1b.output, self.mb1b.input)
        nengo.Connection(self.mb1a.output, self.mb1)
        nengo.Connection(self.mb1b.output, self.mb1)

        # Define network inputs and outputs
        ### TODO: Fix this! (update to include selector and what not)
        self.input = self.mem_in
        self.output = self.mb1

        ### TODO: add selector (to direct info flow to MB units) & other
        #         MB units (mb2, mb3, mbcnt, mbave)

    def connect_from_vision(self, vision_module):
        nengo.Connection(vision_module.am_utilities[item_mb_gate_sp_inds],
                         self.mb1a.gate,
                         transform=[[1] * len(item_mb_gate_sp_inds)])
        nengo.Connection(vision_module.neg_attention,
                         self.mb1a.gate, transform=-1, synapse=0.01)
        nengo.Connection(vision_module.am_utilities[item_mb_gate_sp_inds],
                         self.mb1b.gate,
                         transform=[[1] * len(item_mb_gate_sp_inds)])
        nengo.Connection(vision_module.neg_attention,
                         self.mb1b.gate, transform=-1, synapse=0.01)

        nengo.Connection(vision_module.am_utilities[item_mb_rst_sp_inds],
                         self.mb1a.reset,
                         transform=[[1] * len(item_mb_rst_sp_inds)])
        nengo.Connection(vision_module.am_utilities[item_mb_rst_sp_inds],
                         self.mb1b.reset,
                         transform=[[1] * len(item_mb_rst_sp_inds)])

    def connect_from_encoding(self, enc_module):
        nengo.Connection(enc_module.enc_output, self.mem_in)
