from warnings import warn

import nengo
from nengo.networks import EnsembleArray
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import AssociativeMemory as AM

from ..config import cfg
# from ..vocabs import


class TransformationSystem(Module):
    def __init__(self):
        super(TransformationSystem, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(output=1)

        self.cconv1 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)
        self.cconv2 = cfg.make_cir_conv(input_magnitude=cfg.trans_cconv_radius)

        self.select_cc1a = cfg.make_selector(2)
        self.select_cc1b = cfg.make_selector(5)
        self.route_cc1c = cfg.make_router(4)

        self.select_cc2a = cfg.make_selector(2)
        self.select_cc2b = cfg.make_selector(2)
        self.select_cc2c = cfg.make_selector(2)
        self.route_cc2c = cfg.make_router(1)

        self.mb1_in = nengo.Node(size_in=cfg.sp_dim)
        self.mb2_in = nengo.Node(size_in=cfg.sp_dim)
        self.mb3_in = nengo.Node(size_in=cfg.sp_dim)
        self.mbave_in = nengo.Node(size_in=cfg.sp_dim)

    def setup_connections(self, parent_net):
        pass
