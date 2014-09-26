import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.distributions import Uniform
from nengo.utils.distributions import Choice

from .._spa import AssociativeMemory as AM
from .._networks import CircularConvolution as CConv

from ..config import cfg
from ..vocabs import item_vocab
from ..vocabs import mtr_vocab
from ..vocabs import dec_out_sel_sp_vecs
from ..vision.lif_vision import am_vis_sps


class InfoDecoding(Module):
    def __init__(self):
        super(InfoDecoding, self).__init__()

        # MB x POS~
        self.item_dcconv = CConv(cfg.n_neurons_cconv, cfg.sp_dim,
                                 invert_b=True)

        # Decoding associative memory
        self.dec_am = AM(item_vocab, mtr_vocab, wta_output=True,
                         inhibitable=True, inhibit_scale=5)
        nengo.Connection(self.item_dcconv.output, self.dec_am.input)

        # Transform from visual WM to motor semantic pointer [for copy drawing
        # task]
        ### TODO: Replace with actual transformation matrix
        self.vis_transform = AM(am_vis_sps[:len(mtr_vocab.keys), :],
                                mtr_vocab, wta_output=True,
                                inhibitable=True, inhibit_scale=5)

        # Decoding output selector (selects between decoded from item WM or
        # transformed from visual WM)
        self.select_out = nengo.Ensemble(cfg.n_neurons_ens, 1)
        select_am = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                   intercepts=Uniform(0.5, 1),
                                   encoders=Choice([[1]]))
        select_vis = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                    intercepts=Uniform(0.5, 1),
                                    encoders=Choice([[1]]))
        nengo.Connection(self.select_out, select_am)
        nengo.Connection(self.select_out, select_vis, function=lambda x: 1 - x)
        nengo.Connection(select_am, self.dec_am.inhibit)
        nengo.Connection(select_vis, self.vis_transform.inhibit)

        ## DEBUG
        self.select_am = select_am
        self.select_vis = select_vis

        output = nengo.Node(size_in=cfg.mtr_dim)
        nengo.Connection(self.dec_am.output, output)
        nengo.Connection(self.vis_transform.output, output)

        # Define network inputs and outputs
        self.dec_input = self.item_dcconv.A
        self.pos_input = self.item_dcconv.B
        self.vis_input = self.vis_transform.input
        self.dec_output = output

    def connect_from_vision(self, vision_module):
        nengo.Connection(vision_module.mb_output, self.vis_input)

    def connect_from_prodsys(self, prodsys_module):
        nengo.Connection(prodsys_module.task, self.select_out,
                         transform=np.matrix(dec_out_sel_sp_vecs))

    def connect_from_encoding(self, enc_module):
        nengo.Connection(enc_module.pos_output, self.pos_input)

    def connect_from_memory(self, mem_module):
        nengo.Connection(mem_module.output, self.dec_input)
