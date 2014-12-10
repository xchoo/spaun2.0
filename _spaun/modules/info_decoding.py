import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module

from .._spa import AssociativeMemory as AM

from ..vision.lif_vision import am_vis_sps
from ..config import cfg
from ..vocabs import item_vocab, mtr_vocab, mtr_unk_vocab
from ..vocabs import dec_out_sel_sp_vecs, dec_pos_gate_sp_vecs


class InfoDecoding(Module):
    def __init__(self):
        super(InfoDecoding, self).__init__()

        bias_node = nengo.Node(output=1)

        # MB x POS~
        self.item_dcconv = cfg.make_cir_conv(invert_b=True,
                                             radius=cfg.dcconv_radius)

        # Decoding associative memory
        self.dec_am = AM(item_vocab, mtr_vocab, wta_output=True,
                         inhibitable=True, inhibit_scale=3,
                         wta_inhibit_scale=3.5,
                         threshold=cfg.dec_am_min_thresh,
                         threshold_output=True)
        nengo.Connection(self.item_dcconv.output, self.dec_am.input,
                         synapse=0.01)

        am_gated_ens = cfg.make_ens_array(n_ensembles=cfg.mtr_dim)
        nengo.Connection(self.dec_am.output, am_gated_ens.input)

        # Top 2 am utilities difference calculation
        dec_am2 = AM(item_vocab, item_vocab, wta_output=True,
                     inhibitable=True, inhibit_scale=3,
                     wta_inhibit_scale=3.5, threshold=0.0,
                     threshold_output=True)
        dec_am2.add_input('dec_am_utils', np.eye(len(item_vocab.keys)) * -3)
        nengo.Connection(self.dec_am.thresholded_utilities,
                         dec_am2.dec_am_utils)
        nengo.Connection(self.item_dcconv.output, dec_am2.input, synapse=0.01)

        util_diff = cfg.make_thresh_ens(cfg.dec_am_min_diff)
        # util_diff = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(self.dec_am.utilities, util_diff,
                         transform=[[1] * len(item_vocab.keys)], synapse=0.01)
        nengo.Connection(dec_am2.utilities, util_diff,
                         transform=[[-1] * len(item_vocab.keys)], synapse=0.01)

        util_diff_neg = cfg.make_thresh_ens(1 - cfg.dec_am_min_diff)
        nengo.Connection(bias_node, util_diff_neg)
        # nengo.Connection(util_diff, util_diff_neg, transform=-1,
        nengo.Connection(util_diff, util_diff_neg, transform=-2,
                         synapse=0.01)
        nengo.Connection(self.dec_am.inhibit, util_diff_neg, transform=-2)

        util_diff_thresh = cfg.make_thresh_ens()    # Clean util_diff signal
        nengo.Connection(bias_node, util_diff_thresh)
        nengo.Connection(util_diff_neg, util_diff_thresh, transform=-2,
                         synapse=0.01)

        # Inhibit decoding associative memory when task != (DEC or DECW)
        self.dec_am_task_inhibit = cfg.make_thresh_ens()
        nengo.Connection(bias_node, self.dec_am_task_inhibit)
        nengo.Connection(self.dec_am_task_inhibit, self.dec_am.inhibit,
                         synapse=0.01)
        nengo.Connection(self.dec_am_task_inhibit, dec_am2.inhibit,
                         synapse=0.01)

        # Transform from visual WM to motor semantic pointer [for copy drawing
        # task]
        ### TODO: Replace with actual transformation matrix
        from ..vision.lif_vision import scales_data
        am_threshold = 0.5 * scales_data
        self.vis_transform = AM(am_vis_sps[:len(mtr_vocab.keys), :],
                                mtr_vocab, wta_output=True,
                                threshold=am_threshold[:len(mtr_vocab.keys)],
                                threshold_output=True,
                                inhibitable=True, inhibit_scale=3)

        # Decoding output selector (selects between decoded from item WM or
        # transformed from visual WM)
        self.select_out = nengo.Ensemble(cfg.n_neurons_ens, 1)
        inhibit_am = cfg.make_thresh_ens()
        inhibit_vis = cfg.make_thresh_ens()
        nengo.Connection(self.select_out, inhibit_am)
        nengo.Connection(self.select_out, inhibit_vis,
                         function=lambda x: 1 - x)
        for ens in am_gated_ens.ensembles:
            nengo.Connection(inhibit_am, ens.neurons,
                             transform=[[-3]] * ens.n_neurons)
        nengo.Connection(inhibit_vis, self.vis_transform.inhibit)

        # Decoding POS mem block gate signal generation (from motor system)
        self.pos_mb_gate_bias = cfg.make_thresh_ens()
        self.pos_mb_gate_sig = cfg.make_thresh_ens(0.1)

        # Use pos mb gate signal as inhibition for dec_am2 assoc mem
        # nengo.Connection(self.pos_mb_gate_sig, dec_am2.inhibit,
        #                  synapse=0.01)

        self.dec_am_inhibit = cfg.make_thresh_ens(0.1)
        # nengo.Connection(self.dec_am_inhibit, self.dec_am.inhibit,
        #                  synapse=0.01)
        # nengo.Connection(self.dec_am_inhibit, dec_am2.inhibit,
        #                  synapse=0.01)

        # Delay inhibition to dec_am and vis_transform to allow recall_mb
        # to capture data
        dec_am_inhibit_delay = cfg.make_thresh_ens()
        nengo.Connection(self.dec_am_inhibit, dec_am_inhibit_delay,
                         synapse=0.01)
        nengo.Connection(dec_am_inhibit_delay, self.dec_am.inhibit,
                         synapse=0.01)
        nengo.Connection(dec_am_inhibit_delay, dec_am2.inhibit,
                         synapse=0.01)
        nengo.Connection(dec_am_inhibit_delay, self.vis_transform.inhibit,
                         synapse=0.01)

        # Free recall decoding system
        self.dec_am.add_output('item_output', item_vocab)
        recall_mb = cfg.make_mem_block(gate_mode=1)
        nengo.Connection(recall_mb.output, recall_mb.input)
        nengo.Connection(self.dec_am.item_output, recall_mb.input,
                         synapse=0.01)
        nengo.Connection(self.dec_am_task_inhibit, recall_mb.reset)
        nengo.Connection(self.pos_mb_gate_sig, recall_mb.gate,
                         transform=10)

        self.dec_am_fr = AM(item_vocab, wta_output=True,
                            inhibitable=True, inhibit_scale=5,
                            wta_inhibit_scale=3.5,
                            threshold=cfg.dec_fr_min_thresh,
                            threshold_output=True)
        nengo.Connection(recall_mb.output, self.dec_am_fr.input,
                         transform=-1)
        nengo.Connection(self.pos_mb_gate_sig, self.dec_am_fr.inhibit,
                         synapse=0.01)

        # Add output of free recall am as a small bias to dec_am
        nengo.Connection(self.dec_am_fr.output, self.dec_am.input,
                         transform=cfg.dec_am_min_diff)

        # Output classification (know, don't know, list end) stuff
        # - Logic:
        #     - KNOW if
        #         (dec_am.utils > cfg.dec_am_min_thresh &&
        #          util_diff > cfg.dec_am_min_diff)
        #     - DON'T KNOW if
        #         ((dec_am_fr.util > cfg.dec_fr_min_thresh &&
        #           dec_am.util < cfg.dec_am_min_thresh) ||
        #          (util_diff < cfg.dec_am_min_diff &&
        #           dec_am.util > cfg.dec_am_min_thresh))
        #     - STOP if
        #         (dec_am.util < cfg.dec_am_min_thresh &&
        #          dec_am_fr.util < cfg.dec_fr_min_thresh)
        dec_am_y = self.dec_am.thresholded_utilities
        dec_am_n = self.dec_am.default_output_thresholded_utility

        dec_am_diff_y = util_diff_thresh

        fr_am_n = self.dec_am_fr.default_output_thresholded_utility

        output_know = cfg.make_thresh_ens(0.55)
        output_unk = cfg.make_thresh_ens(0.80)
        output_stop = cfg.make_thresh_ens(0.75)

        nengo.Connection(dec_am_y, output_know,
                         transform=[[0.5] * self.dec_am.num_items],
                         synapse=0.01)
        nengo.Connection(dec_am_diff_y, output_know, transform=0.5,
                         synapse=0.01)

        nengo.Connection(bias_node, output_unk)
        nengo.Connection(output_know, output_unk, transform=-5,
                         synapse=0.01)
        nengo.Connection(self.pos_mb_gate_sig, output_unk, transform=-2,
                         synapse=0.01)
        nengo.Connection(self.dec_am.inhibit, output_unk, transform=-2,
                         synapse=0.01)

        nengo.Connection(dec_am_n, output_stop, transform=0.5, synapse=0.01)
        nengo.Connection(fr_am_n, output_stop, transform=0.5, synapse=0.01)
        nengo.Connection(self.pos_mb_gate_sig, output_stop, transform=-2,
                         synapse=0.01)

        # Output "UNK" vector to motor system.
        # Inhibited if output is KNOW, or if not using dec_am outputs
        # (i.e. during copy drawing)
        bias_unk_vec = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(bias_node, bias_unk_vec, synapse=None)
        nengo.Connection(self.dec_am.inhibit, bias_unk_vec.neurons,
                         transform=[[-3]] * bias_unk_vec.n_neurons,
                         synapse=0.01)
        nengo.Connection(self.pos_mb_gate_sig, bias_unk_vec.neurons,
                         transform=[[-3]] * bias_unk_vec.n_neurons,
                         synapse=0.01)
        nengo.Connection(output_know, bias_unk_vec.neurons,
                         transform=[[-3]] * bias_unk_vec.n_neurons,
                         synapse=0.01)
        nengo.Connection(output_stop, bias_unk_vec.neurons,
                         transform=[[-3]] * bias_unk_vec.n_neurons,
                         synapse=0.01)

        # Inhibit dec_am output (through am_gated_ens) if output is DON'T KNOW
        for e in am_gated_ens.ensembles:
            nengo.Connection(output_unk, e.neurons,
                             transform=[[-3]] * e.n_neurons)

        ############################## DEBUG ##################################
        self.select_am = inhibit_am
        self.select_vis = inhibit_vis

        self.am_out = nengo.Node(size_in=cfg.mtr_dim)
        self.vt_out = nengo.Node(size_in=cfg.mtr_dim)
        nengo.Connection(self.dec_am.output, self.am_out, synapse=None)
        nengo.Connection(self.vis_transform.output, self.vt_out, synapse=None)

        self.am_utils = self.dec_am.utilities
        self.am2_utils = dec_am2.utilities
        self.util_diff = util_diff

        self.am_th_utils = self.dec_am.thresholded_utilities
        self.fr_th_utils = self.dec_am_fr.thresholded_utilities

        self.recall_mb = recall_mb

        self.debug_task = nengo.Node(size_in=1)

        self.output_know = output_know
        self.output_unk = output_unk

        self.util_diff_neg = util_diff_neg

        ############################ END DEBUG ################################

        # Define network inputs and outputs
        self.dec_input = self.item_dcconv.A
        self.pos_input = self.item_dcconv.B
        self.vis_input = self.vis_transform.input
        self.output_stop = output_stop

        self.dec_output = nengo.Node(size_in=cfg.mtr_dim)
        nengo.Connection(am_gated_ens.output, self.dec_output)
        nengo.Connection(self.vis_transform.output, self.dec_output)
        nengo.Connection(bias_unk_vec, self.dec_output,
                         transform=np.matrix(mtr_unk_vocab['UNK'].v).T)

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            nengo.Connection(p_net.vis.mb_output, self.vis_input)
        else:
            warn("InfoDecoding Module - Could not connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            nengo.Connection(p_net.ps.task, self.select_out,
                             transform=[dec_out_sel_sp_vecs * 1.0])
            nengo.Connection(p_net.ps.task, self.pos_mb_gate_bias,
                             transform=[dec_pos_gate_sp_vecs * 1.0])
            nengo.Connection(p_net.ps.task, self.dec_am_task_inhibit,
                             transform=[dec_pos_gate_sp_vecs * -1.0])

            ####### DEBUG ########
            nengo.Connection(p_net.ps.task, self.debug_task,
                             transform=[dec_pos_gate_sp_vecs * 1.0])
        else:
            warn("InfoDecoding Module - Could not connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.pos_output, self.pos_input)
            nengo.Connection(self.pos_mb_gate_bias, p_net.enc.pos_mb.gate,
                             transform=4, synapse=0.01)
            nengo.Connection(self.pos_mb_gate_sig, p_net.enc.pos_mb.gate,
                             transform=-4, synapse=0.01)
        else:
            warn("InfoDecoding Module - Could not connect from 'enc'")

        # Set up connections from memory module
        if hasattr(p_net, 'mem'):
            nengo.Connection(p_net.mem.output, self.dec_input,
                             transform=cfg.dcconv_item_in_scale)
            nengo.Connection(p_net.mem.output, self.dec_am_fr.input,
                             transform=cfg.dec_fr_item_in_scale)
        else:
            warn("InfoDecoding Module - Could not connect from 'mem'")

        # Set up connections from motor module
        if hasattr(p_net, 'mtr'):
            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.pos_mb_gate_sig,
                             synapse=0.005, transform=5,
                             function=lambda x: x > 0.1)
            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.pos_mb_gate_sig,
                             synapse=0.08, transform=-10,
                             function=lambda x: x > 0.1)

            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.dec_am_inhibit,
                             synapse=0.005, transform=5,
                             function=lambda x: x > 0.1)
            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.dec_am_inhibit,
                             synapse=0.01, transform=-10,
                             function=lambda x: x > 0.1)
        else:
            warn("InfoDecoding Module - Could not connect from 'mtr'")
