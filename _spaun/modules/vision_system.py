from warnings import warn
import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import DetectChange

from ..configurator import cfg
from ..vocabulator import vocab
from .stimulus import stim_func_vocab
from .stim import stim_data
from .vision import vis_data, VisionNet, VisionNetClassifier


class VisionSystem(Module):
    def __init__(self, label="Vision Sys", seed=None, add_to_container=None,
                 vis_net=None, detect_net=None,
                 vis_net_cfg=None, vis_net_neuron_type=None):
        super(VisionSystem, self).__init__(label, seed, add_to_container)
        if vis_net_cfg is None:
            vis_net_cfg = vis_data
        self.init_module(vis_net, detect_net, vis_net_cfg, vis_net_neuron_type)

    @with_self
    def init_module(self, vis_net, detect_net, vis_net_cfg,
                    vis_net_neuron_type):
        # Make LIF vision network
        if vis_net is None:
            vis_net = VisionNet(vis_net_cfg, stim_data,
                                net_neuron_type=vis_net_neuron_type)
        self.vis_net = vis_net

        # Make network to detect changes in visual input stream
        # Limit detection network dimensionality to avoid massive neuron count
        image_dim = stim_data.images_data_dimensions
        detect_net_max_dim = min(cfg.vis_detect_dim, image_dim)
        detect_net_inds = np.random.permutation(image_dim)[:detect_net_max_dim]

        if detect_net is None:
            detect_net = \
                DetectChange(dimensions=detect_net_max_dim,
                             n_neurons=cfg.n_neurons_ens)
        self.detect_change_net = detect_net
        nengo.Connection(self.vis_net.raw_output[detect_net_inds],
                         self.detect_change_net.input, synapse=None)

        # Make associative memory to map visual image semantic pointers to
        # visual conceptual semantic pointers
        self.vis_classify = VisionNetClassifier(vis_net_cfg,
                                                vocab.vis_main.vectors)
        nengo.Connection(self.vis_net.to_classify_output,
                         self.vis_classify.input, synapse=0.005)
        # nengo.Connection(self.detect_change_net.output,
        #                  self.vis_classify.inhibit, transform=3, synapse=0.005)

        detect_change_net_delay = cfg.make_thresh_ens_net(0.80)
        nengo.Connection(self.detect_change_net.output,
                         detect_change_net_delay.input)
        nengo.Connection(detect_change_net_delay.output,
                         self.vis_classify.inhibit, synapse=0.01, transform=3)

        # Visual memory block (for the visual semantic pointers - top layer of
        #                      vis_net)
        self.vis_mem = cfg.make_memory(n_neurons=cfg.n_neurons_mb * 2,
                                       dimensions=vocab.vis_dim,
                                       ens_dimensions=1,
                                       make_ens_func=cfg.make_ens_array,
                                       radius=vis_net_cfg.sps_element_scale)
        nengo.Connection(self.vis_net.to_mem_output, self.vis_mem.input,
                         synapse=0.02)

        # bias_node = nengo.Node(1, label='Bias')
        # nengo.Connection(bias_node, self.vis_mem.gate)
        # vis_mem_gate_sp_vecs = vocab.main.parse('+'.join(vocab.num_sp_strs)).v
        # nengo.Connection(self.vis_classify.output, self.vis_mem.gate,
        #                  transform=[cfg.mb_neg_gate_scale *
        #                             vis_mem_gate_sp_vecs])
        # # vis_mem holds value when gate == 1, so have standard 1-x gating
        # # signal (i.e. allow values when gate is 1 - <vis_mem_gate_sp_vecs>)
        # # circuit here.

        vis_mem_no_gate_sp_vecs = \
            vocab.main.parse('+'.join(vocab.ps_task_vis_sp_strs +
                                      vocab.misc_vis_sp_strs)).v
        nengo.Connection(self.vis_classify.output, self.vis_mem.gate,
                         transform=[cfg.mb_gate_scale *
                                    vis_mem_no_gate_sp_vecs])
        # vis_mem holds value when gate == 1, so just have to connect to gate
        # (i.e. don't gate when vis_classify.output ==
        # <vis_mem_no_gate_sp_vecs>).

        nengo.Connection(self.detect_change_net.output, self.vis_mem.gate,
                         transform=5, synapse=0.02)

        # Visual memory (for the visual concept semantic pointers - out
        #                of the AM)
        self.vis_main_mem = \
            cfg.make_memory(dimensions=vocab.sp_dim, n_neurons=100,
                            represent_identity=False, ens_dimensions=1,
                            make_ens_func=cfg.make_spa_ens_array)
        nengo.Connection(self.vis_classify.output, self.vis_main_mem.input,
                         synapse=0.02)
        nengo.Connection(self.detect_change_net.output,
                         self.vis_main_mem.gate, transform=5)

        # Define network input and outputs
        self.input = self.vis_net.input
        self.output = self.vis_classify.output
        self.mb_output = self.vis_mem.output
        self.neg_attention = self.detect_change_net.output

        # Define module inputs and outputs
        self.outputs = dict(default=(self.output, vocab.vis_main),
                            mem=(self.vis_main_mem.output, vocab.vis_main))

        # ######################## DEBUG PROBES ###############################
        self.vis_out = self.vis_net.to_classify_output
        self.vis_classify_utilities = self.vis_classify.output_utilities

        # def cleanup_func(t, x, vectors):
        #     return vectors[np.argmax(np.dot(x, vectors.T)), :]

        # def rmse_func(t, x, dim):
        #     v1 = x[:dim]
        #     v2 = x[dim:]
        #     return np.sqrt(np.sum((v1 - v2) ** 2))

        # def diff_func(t, x, dim):
        #     v1 = x[:dim]
        #     v2 = x[dim:]
        #     return np.array(v1 - v2)

        # self.cleanup_node = nengo.Node(
        #     size_in=vocab.sp_dim,
        #     output=lambda t, x, vectors=vocab.vis_main.vectors:
        #     cleanup_func(t, x, vectors))
        # nengo.Connection(self.vis_main_mem.output, self.cleanup_node,
        #                  synapse=0.01)

        # self.rmse_node = nengo.Node(
        #     size_in=vocab.sp_dim * 2,
        #     output=lambda t, x, dim=vocab.sp_dim: rmse_func(t, x, dim))
        # nengo.Connection(self.vis_main_mem.output,
        #                  self.rmse_node[:vocab.sp_dim], synapse=0.01)
        # nengo.Connection(self.cleanup_node, self.rmse_node[vocab.sp_dim:],
        #                  synapse=0.01)

        # self.diff_node = nengo.Node(
        #     size_in=vocab.sp_dim * 2,
        #     output=lambda t, x, dim=vocab.sp_dim: diff_func(t, x, dim))
        # nengo.Connection(self.vis_main_mem.output,
        #                  self.diff_node[:vocab.sp_dim], synapse=0.01)
        # nengo.Connection(self.cleanup_node, self.diff_node[vocab.sp_dim:],
        #                  synapse=0.01)

    def setup_connections(self, parent_net):
        # Set up connections from stimulus module
        if hasattr(parent_net, 'stim'):
            nengo.Connection(parent_net.stim.output, self.input)
        else:
            warn("Vision Module - Cannot connect from 'stim'")


class VisionSystemDummy(VisionSystem):
    def __init__(self, label="Dummy Vision Sys", seed=None,
                 add_to_container=None,
                 vis_net=None, detect_net=None,
                 vis_net_cfg=vis_data, vis_net_neuron_type=None, **args):
        super(VisionSystemDummy, self).__init__(label, seed, add_to_container,
                                                self.dummy_lif_vis_net(),
                                                self.dummy_detect_net(),
                                                vis_net_cfg, **args)

        # Indicate to the transform system that we are using a dummy vision
        # system
        vocab.vis_dim = -vocab.sp_dim

    def dummy_lif_vis_net(self):
        with nengo.Network(label="Dummy LIF Vision") as net:
            net.input = nengo.Node(size_in=stim_data.images_data_dimensions,
                                   label='Input')
            net.output = nengo.Node(output=stim_func_vocab,
                                    label='Dummy LIF Vision Out')
            net.raw_output = net.output
        return net

    def dummy_detect_net(self):
        return DetectChange(None, dimensions=vocab.sp_dim,
                            item_magnitude=(cfg.get_optimal_sp_radius() / 2.0))
