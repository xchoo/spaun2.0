from warnings import warn

import nengo
from nengo.spa import Vocabulary
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import DetectChange
from .._spa import MemoryBlock as MB

from ..configurator import cfg
from ..vocabulator import vocab
from .stimulus import stim_func_vocab
from .vision.lif_vision import LIFVision as LIFVisionNet
from .vision.data import vis_data


class VisionSystem(Module):
    def __init__(self, label="Vision Sys", seed=None, add_to_container=None,
                 vis_net=None, detect_net=None, vis_sps=None,
                 vis_sps_scale=None, vis_net_neuron_type=None):
        super(VisionSystem, self).__init__(label, seed, add_to_container)
        if vis_sps is None:
            vis_sps = vis_data.sps
        if vis_sps_scale is None:
            vis_sps_scale = vis_data.sps_scale
        self.init_module(vis_net, detect_net, vis_sps, vis_sps_scale,
                         vis_net_neuron_type)

    @with_self
    def init_module(self, vis_net, detect_net, vis_sps, vis_sps_scale,
                    vis_net_neuron_type):
        # Make LIF vision network
        if vis_net is None:
            vis_net = LIFVisionNet(net_neuron_type=vis_net_neuron_type)
        self.vis_net = vis_net

        # Make network to detect changes in visual input stream
        if detect_net is None:
            detect_net = \
                DetectChange(dimensions=vis_data.images_data_dimensions,
                             n_neurons=cfg.n_neurons_ens)
        self.detect_change_net = detect_net
        nengo.Connection(self.vis_net.raw_output, self.detect_change_net.input,
                         synapse=None)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.am = cfg.make_assoc_mem(vis_sps, vocab.vis_main.vectors,
                                     threshold=vis_data.am_threshold,
                                     inhibitable=True)
        nengo.Connection(self.vis_net.output, self.am.input, synapse=0.005)
        nengo.Connection(self.detect_change_net.output, self.am.inhibit,
                         transform=3, synapse=0.005)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        vocab.vis_dim = vis_sps.shape[1]
        vis_sp_vocab = Vocabulary(vocab.vis_dim)  # TODO: FIX THIS?

        self.vis_mb = MB(cfg.n_neurons_mb * 2, vocab.vis_dim, gate_mode=2,
                         vocab=vis_sp_vocab, radius=vis_sps_scale,
                         **cfg.mb_config)

        vis_mb_gate_sp_vecs = vocab.main.parse('+'.join(vocab.num_sp_strs)).v
        nengo.Connection(self.am.cleaned_output, self.vis_mb.gate,
                         transform=[cfg.mb_gate_scale * vis_mb_gate_sp_vecs])
        nengo.Connection(self.vis_net.output, self.vis_mb.input,
                         transform=vis_data.amp, synapse=0.03)
        nengo.Connection(self.detect_change_net.output,
                         self.vis_mb.gate, transform=-1, synapse=0.01)

        # Define network input and outputs
        self.input = self.vis_net.input
        self.output = self.am.cleaned_output
        self.mb_output = self.vis_mb.output
        self.neg_attention = self.detect_change_net.output

        # Define module inputs and outputs
        self.outputs = dict(default=(self.output, vocab.vis_main))

        # ######################## DEBUG PROBES ###############################
        self.vis_out = self.vis_net.output
        self.am_utilities = self.am.cleaned_output_utilities

    def setup_connections(self, parent_net):
        # Set up connections from stimulus module
        if hasattr(parent_net, 'stim'):
            nengo.Connection(parent_net.stim.output, self.input)
        else:
            warn("Vision Module - Cannot connect from 'stim'")


class VisionSystemDummy(VisionSystem):
    def __init__(self, label="Dummy Vision Sys", seed=None, add_to_container=None):
        super(VisionSystemDummy, self).__init__(label, seed, add_to_container)

        # Indicate to the transform system that we are using a dummy vision
        # system
        vocab.vis_dim = -vocab.sp_dim

    def dummy_lif_vis_net(self):
        with nengo.Network(label="Dummy LIF Vision") as net:
            net.input = nengo.Node(size_in=vis_data.images_data_dimensions,
                                   label='Input')
            net.output = nengo.Node(output=stim_func_vocab,
                                    label='Dummy LIF Vision Out')
            net.raw_output = net.output
        return net

    def dummy_detect_net(self):
        return DetectChange(None, dimensions=vocab.sp_dim,
                            item_magnitude=(cfg.get_optimal_sp_radius() / 2.0))
