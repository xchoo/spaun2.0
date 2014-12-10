from warnings import warn

import nengo
from nengo.spa.module import Module

from ..config import cfg
from ..vocabs import item_mb_gate_sp_inds
from ..vocabs import item_mb_rst_sp_inds


class WorkingMemory(Module):
    def __init__(self):
        super(WorkingMemory, self).__init__()

        # Memory input node
        self.mem_in = nengo.Node(size_in=cfg.sp_dim, label='WM Module In Node')

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1 = nengo.Node(size_in=cfg.sp_dim, label='MB1 In Node')
        self.mb1a = cfg.make_mem_block(label='MB1A (Rehearsal)')
        self.mb1b = cfg.make_mem_block(fdbk_transform=cfg.mb_decay_val,
                                       label='MB1B (Decay)')
        nengo.Connection(self.mem_in, self.mb1a.input)
        nengo.Connection(self.mem_in, self.mb1b.input,
                         transform=cfg.mb_decaybuf_input_scale)
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

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            ####### MB1 ########
            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb1a.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb1a.gate, transform=-1, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_gate_sp_inds],
                             self.mb1b.gate,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_gate_sp_inds)])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mb1b.gate, transform=-1, synapse=0.01)

            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb1a.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])
            nengo.Connection(p_net.vis.am_utilities[item_mb_rst_sp_inds],
                             self.mb1b.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(item_mb_rst_sp_inds)])

            ####### MB2 ########
        else:
            warn("WorkingMemory Module - Cannot connect from 'vis'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.enc_output, self.mem_in)
        else:
            warn("WorkingMemory Module - Cannot connect from 'enc'")
