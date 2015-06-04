from warnings import warn

import nengo
from nengo.spa import Vocabulary
from nengo.spa.module import Module
from nengo.utils.network import with_self
from nengo.dists import Uniform, Choice

from .._spa import MemoryBlock as MB

from ..config import cfg
from ..vocabs import vocab, vis_vocab
from ..vocabs import item_mb_gate_sp_inds
from .stimulus import stim_func_vocab
from ..vision.lif_vision import LIFVision as LIFVisionNet
from ..vision.lif_vision import vis_sps_scale as lif_vis_sps_scale
from ..vision.lif_vision import am_vis_sps, am_threshold
from ..vision.lif_vision import images_data_dimensions


def DetectChangeNet(net=None, dim=images_data_dimensions, diff_scale=0.2,
                    item_magnitude=1):
    # item_magnitude: expected magnitude of one pixel in the input image

    if net is None:
        net = nengo.Network(label="Detect Change Network")

    with net:
        net.input = nengo.Node(size_in=dim)

        # Negative attention signal generation. Generates a high valued signal
        # when the input is changing or when there is nothing being presented
        # to the visual system
        input_diff = nengo.networks.EnsembleArray(nengo.Default, dim,
                                                  label='input differentiator',
                                                  intercepts=Uniform(0.1, 1))
        input_diff.add_output('abs', lambda x: abs(x))
        nengo.Connection(net.input, input_diff.input, synapse=0.005,
                         transform=1.0 / item_magnitude)
        nengo.Connection(net.input, input_diff.input, synapse=0.020,
                         transform=-1.0 / item_magnitude)

        #######################################################################
        net.output = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                    intercepts=Uniform(0.5, 1),
                                    encoders=Choice([[1]]))
        nengo.Connection(input_diff.abs, net.output, synapse=0.005,
                         transform=[[diff_scale] * dim])

        item_detect = nengo.Ensemble(nengo.Default, 1)
        nengo.Connection(net.input, item_detect, synapse=0.005,
                         transform=[[1.0 / item_magnitude] * dim])
        nengo.Connection(item_detect, net.output, synapse=0.005,
                         function=lambda x: 1 - abs(x))

        blank_detect = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                      intercepts=Uniform(0.7, 1),
                                      encoders=Choice([[1]]))
        nengo.Connection(item_detect, blank_detect, synapse=0.005,
                         function=lambda x: 1 - abs(x))
        #######################################################################

        # Delay ensemble needed to smooth out transition from blank to
        # change detection
        blank_detect_delay = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                            intercepts=Uniform(0.1, 1),
                                            encoders=Choice([[1]]))
        nengo.Connection(blank_detect, blank_detect_delay, synapse=0.03)
        nengo.Connection(blank_detect, net.output, synapse=0.01,
                         transform=2)

        # ### DEBUG ####
        net.input_diff = input_diff.output
        net.item_detect = item_detect
        net.blank_detect = blank_detect

    return net


class VisionSystem(Module):
    def __init__(self, label="Vision Sys", seed=None, add_to_container=None,
                 vis_net=LIFVisionNet, detect_net=DetectChangeNet,
                 vis_sps=am_vis_sps, vis_sps_scale=lif_vis_sps_scale):
        super(VisionSystem, self).__init__(label, seed, add_to_container)
        self.init_module(vis_net, detect_net, vis_sps, vis_sps_scale)

    @with_self
    def init_module(self, vis_net, detect_net, vis_sps, vis_sps_scale):
        # Make LIF vision network
        self.vis_net = vis_net()

        # Make network to detect changes in visual input stream
        self.detect_change_net = detect_net()
        nengo.Connection(self.vis_net.raw_output, self.detect_change_net.input,
                         synapse=None)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.am = cfg.make_assoc_mem(vis_sps, vis_vocab.vectors,
                                     threshold=am_threshold,
                                     inhibitable=True, inhibit_scale=5)
        nengo.Connection(self.vis_net.output, self.am.input, synapse=0.005)
        nengo.Connection(self.detect_change_net.output, self.am.inhibit,
                         synapse=0.005)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        cfg.vis_dim = vis_sps.shape[1]
        vis_sp_vocab = Vocabulary(cfg.vis_dim)  # TODO: FIX THIS?
        self.vis_mb = MB(cfg.n_neurons_mb * 2, cfg.vis_dim, gate_mode=2,
                         vocab=vis_sp_vocab, radius=vis_sps_scale,
                         **cfg.mb_config)
        nengo.Connection(self.am.thresholded_utilities[item_mb_gate_sp_inds],
                         self.vis_mb.gate,
                         transform=[[cfg.mb_gate_scale] *
                                    len(item_mb_gate_sp_inds)])
        nengo.Connection(self.vis_net.output, self.vis_mb.input, synapse=0.01)
        nengo.Connection(self.detect_change_net.output,
                         self.vis_mb.gate, transform=-1, synapse=0.01)

        # Define network input and outputs
        self.input = self.vis_net.input
        self.output = self.am.output
        self.mb_output = self.vis_mb.output
        self.am_utilities = self.am.thresholded_utilities
        self.neg_attention = self.detect_change_net.output

        # Define module inputs and outputs
        self.outputs = dict(default=(self.output, vocab))

    def setup_connections(self, parent_net):
        # Set up connections from stimulus module
        if hasattr(parent_net, 'stim'):
            nengo.Connection(parent_net.stim.output, self.input)
        else:
            warn("Vision Module - Cannot connect from 'stim'")


class VisionSystemDummy(VisionSystem):
    def __init__(self, label="Dummy Vision Sys", seed=None,
                 add_to_container=None):
        super(VisionSystemDummy, self).__init__(label, seed, add_to_container,
                                                self.dummy_lif_vis_net,
                                                self.dummy_detect_net,
                                                vis_vocab.vectors,
                                                cfg.get_optimal_sp_radius())

        # Indicate to the transform system that we are using a dummy vision
        # system
        cfg.vis_dim = -1

    def dummy_lif_vis_net(self, net=None):
        if net is None:
            net = nengo.Network(label="Dummy LIF Vision")

        with net:
            net.input = nengo.Node(size_in=images_data_dimensions,
                                   label='Input')
            net.output = nengo.Node(output=stim_func_vocab,
                                    label='Dummy LIF Vision Out')
            net.raw_output = net.output
        return net

    def dummy_detect_net(self):
        return DetectChangeNet(None, cfg.sp_dim,
                               item_magnitude=(cfg.get_optimal_sp_radius() /
                                               2.0))
