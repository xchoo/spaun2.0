import nengo
from nengo.spa.module import Module
from nengo.utils.distributions import Uniform
from nengo.utils.distributions import Choice

from .._spa import MemoryBlock as MB
from .._networks import CircularConvolution as CConv

from ..config import cfg
from ..vocabs import vocab
from ..vocabs import pos_vocab
from ..vocabs import item_vocab
from ..vocabs import pos_mb_gate_sp_inds
from ..vocabs import pos_mb_rst_sp_inds


class InfoEncoding(Module):
    def __init__(self):
        super(InfoEncoding, self).__init__()

        # Node that just outputs the INC vector
        self.inc_vec = nengo.Node(output=vocab['INC'].v)
        self.pos1_vec = nengo.Node(output=vocab['POS1'].v)

        # Memory block to store POS vector
        self.pos_mb = MB(cfg.n_neurons_mb, cfg.sp_dim, gate_mode=2)

        # POS x INC
        self.pos_cconv = CConv(cfg.n_neurons_cconv, cfg.sp_dim)
        nengo.Connection(self.pos_mb.output, self.pos_cconv.A)
        nengo.Connection(self.inc_vec, self.pos_cconv.B)

        # Set up reset mechanism for POS vectors
        ### TODO: Put these things inside a selector network??
        pos_inc_gate = nengo.networks.EnsembleArray(cfg.n_neurons_ens,
                                                    cfg.sp_dim)
        pos_pos1_gate = nengo.networks.EnsembleArray(cfg.n_neurons_ens,
                                                     cfg.sp_dim)
        nengo.Connection(pos_inc_gate.output, self.pos_mb.input)
        nengo.Connection(self.pos_cconv.output, pos_inc_gate.input)
        nengo.Connection(pos_pos1_gate.output, self.pos_mb.input)
        nengo.Connection(self.pos1_vec, pos_pos1_gate.input)

        self.pos_rst = nengo.Ensemble(cfg.n_neurons_ens, 1)
        pos_inc = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                 intercepts=Uniform(0.5, 1),
                                 encoders=Choice([[1]]))
        pos_pos1 = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                  intercepts=Uniform(0.5, 1),
                                  encoders=Choice([[1]]))
        nengo.Connection(self.pos_rst, pos_inc)
        nengo.Connection(self.pos_rst, pos_pos1, function=lambda x: 1 - x)
        for e in pos_inc_gate.ensembles:
            nengo.Connection(pos_inc, e.neurons,
                             transform=[[-3]] * e.n_neurons)
        for e in pos_pos1_gate.ensembles:
            nengo.Connection(pos_pos1, e.neurons,
                             transform=[[-3]] * e.n_neurons)

        # POS x ITEM
        self.item_cconv = CConv(cfg.n_neurons_cconv, cfg.sp_dim)
        nengo.Connection(self.pos_mb.output, self.item_cconv.A)

        # Define network inputs and outputs
        self.pos_output = self.pos_mb.output
        self.item_input = self.item_cconv.B
        self.enc_output = self.item_cconv.output

        # Define module inputs and outputs
        self.inputs = dict(default=(self.item_input, item_vocab))
        self.outputs = dict(default=(self.pos_output, pos_vocab))

    def connect_from_vision(self, vision_module):
        nengo.Connection(vision_module.am_utilities[pos_mb_gate_sp_inds],
                         self.pos_mb.gate,
                         transform=[[1] * len(pos_mb_gate_sp_inds)])
        nengo.Connection(vision_module.neg_attention,
                         self.pos_mb.gate, transform=-1, synapse=0.01)

        nengo.Connection(vision_module.am_utilities[pos_mb_rst_sp_inds],
                         self.pos_rst,
                         transform=[[1] * len(pos_mb_rst_sp_inds)])

        nengo.Connection(vision_module.output, self.item_input)
