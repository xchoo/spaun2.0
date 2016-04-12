import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import vocab, item_vocab, mtr_vocab, mtr_unk_vocab
from ..vocabs import dec_out_sel_sp_vecs, dec_pos_gate_dec_sp_vecs
from ..vocabs import dec_pos_gate_task_sp_vecs
from ..vocabs import pos_mb_rst_sp_inds
from ..vocabs import mtr_sp_scale_factor

from .decoding import Serial_Recall_Network, Free_Recall_Network
from .decoding import Visual_Transform_Network, Output_Classification_Network


class InfoDecoding(Module):
    def __init__(self, label="Info Dec", seed=None, add_to_container=None):
        super(InfoDecoding, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(output=1)

        # ---------------------- Inputs and outputs ------------------------- #
        self.items_input = nengo.Node(size_in=cfg.sp_dim)
        self.pos_input = nengo.Node(size_in=cfg.sp_dim)

        # ----------------- Inhibition signal generation -------------------- #
        # Inhibition signal for when TASK != DEC
        self.dec_am_task_inhibit = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, self.dec_am_task_inhibit.input,
                         synapse=None)

        # Generic inhibition signal?
        self.dec_am_inhibit = cfg.make_thresh_ens_net(0.1)

        # ---------- Decoding POS mem block gate signal generation ---------- #
        # Decoding POS mem block gate signal generation (from motor system)
        self.pos_mb_gate_bias = cfg.make_thresh_ens_net(n_neurons=100)
        self.pos_mb_gate_sig = cfg.make_thresh_ens_net(0.3)

        # Bias does ...?
        # Gate signal does ...?

        # Suppress pos_mb gate bias unless task=DEC + dec=FWD|REV|DECI|DECW
        nengo.Connection(bias_node, self.pos_mb_gate_bias.input, transform=-1)

        # -------------------- Serial decoding network ---------------------- #
        serial_decode = Serial_Recall_Network()
        nengo.Connection(self.items_input, serial_decode.items_input,
                         synapse=None)
        nengo.Connection(self.pos_input, serial_decode.pos_input,
                         synapse=None)

        # Inhibitory connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         serial_decode.inhibit, synapse=0.01)

        # ---------------- Free recall decoding network --------------------- #
        free_recall_decode = Free_Recall_Network()
        nengo.Connection(self.items_input, free_recall_decode.items_input)
        nengo.Connection(self.pos_input, free_recall_decode.pos_input)

        # Add output of free recall am as a small bias to dec_am
        nengo.Connection(free_recall_decode.output, serial_decode.items_input,
                         transform=cfg.dec_fr_to_am_scale)

        # Gating connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         free_recall_decode.reset)
        nengo.Connection(self.pos_mb_gate_bias.output, free_recall_decode.gate,
                         transform=2, synapse=0.08)
        #  Why is there such a large synapse here?
        nengo.Connection(self.pos_mb_gate_sig.output, free_recall_decode.gate,
                         transform=-2, synapse=0.01)

        # Inhibitory connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         free_recall_decode.inhibit, synapse=0.01)

        # nengo.Connection(self.pos_mb_gate_sig.output,
        #                  free_recall_decode.inhibit, synapse=0.01)
        # IS THIS STILL NEEDED?
        self.free_recall_decode = free_recall_decode

        # ------------- Visual transform decoding network ------------------- #
        vis_trfm_decode = Visual_Transform_Network()

        # -------- Output classification (know / unknown / stop) system ----- #
        output_classify = Output_Classification_Network()
        nengo.Connection(serial_decode.dec_success,
                         output_classify.sr_utils_y,
                         transform=[[1.0] * serial_decode.dec_am1.n_items],
                         synapse=0.01)
        nengo.Connection(serial_decode.am_utils_diff,
                         output_classify.sr_utils_diff, synapse=0.01)
        nengo.Connection(serial_decode.dec_failure,
                         output_classify.sr_utils_n, synapse=0.03)
        nengo.Connection(free_recall_decode.dec_failure,
                         output_classify.fr_utils_n, synapse=0.03)

        # Output classification inhibitory signals
        # - Inhibit UNK when am's are inhibited.
        # - Inhibit UNK and STOP when pos_gate_sig is HIGH
        #   (i.e. decoding system is doing things)
        nengo.Connection(self.dec_am_inhibit.output,
                         output_classify.output_unk_inhibit, synapse=0.03)
        nengo.Connection(self.pos_mb_gate_sig.output,
                         output_classify.output_unk_inhibit, synapse=0.03)
        nengo.Connection(self.pos_mb_gate_sig.output,
                         output_classify.output_stop_inhibit, synapse=0.03)

        # ----------------------- Output selector --------------------------- #
        # 0: Serial decoder output
        # 1: Copy drawing transform out\put
        # 2: Free recall decoder output
        # 3: UNK mtr SP output
        # 4: NULL (all zeros) output
        self.select_out = cfg.make_selector(5, radius=mtr_sp_scale_factor,
                                            dimensions=cfg.mtr_dim,
                                            threshold_sel_in=True)

        # Connections for sel0 - SR
        nengo.Connection(serial_decode.output, self.select_out.input0)
        nengo.Connection(output_classify.output_know, self.select_out.sel0)
        nengo.Connection(output_classify.output_unk, self.select_out.sel0,
                         transform=-1)

        # Connections for sel1 - Copy Drawing
        nengo.Connection(vis_trfm_decode.output, self.select_out.input1)
        nengo.Connection(output_classify.output_unk, self.select_out.sel1,
                         transform=-1)

        # Connections for sel2 - FR
        # nengo.Connection(free_recall_decode.output, self.select_out.input2)
        nengo.Connection(output_classify.output_unk, self.select_out.sel2,
                         transform=-1)

        # Connections for sel3 - UNK
        nengo.Connection(bias_node, self.select_out.input3,
                         transform=mtr_unk_vocab['UNK'].v[:, None])
        nengo.Connection(output_classify.output_unk, self.select_out.sel3)

        # Connections for sel4 - NULL
        nengo.Connection(output_classify.output_stop, self.select_out.sel4)

        # ------------------------------------------------------------------- #
        # # MB x POS~
        # self.item_dcconv = cfg.make_cir_conv(invert_b=True,
        #                                      input_magnitude=cfg.dcconv_radius)

        # # Decoding associative memory
        # self.dec_am = \
        #     cfg.make_assoc_mem(item_vocab.vectors, mtr_vocab.vectors,
        #                        inhibitable=True, inhibit_scale=3,
        #                        threshold=cfg.dec_am_min_thresh,
        #                        default_output_vector=np.zeros(cfg.mtr_dim))
        # nengo.Connection(self.item_dcconv.output, self.dec_am.input,
        #                  synapse=0.01)

        # am_gated_ens = cfg.make_ens_array(n_ensembles=cfg.mtr_dim)
        # nengo.Connection(self.dec_am.output, am_gated_ens.input)

        # Top 2 am utilities difference calculation
        # dec_am2 = cfg.make_assoc_mem(item_vocab.vectors, item_vocab.vectors,
        #                              inhibitable=True, inhibit_scale=3,
        #                              threshold=0.0)
        # dec_am2.add_input('dec_am_utils', np.eye(len(item_vocab.keys)) * -3)
        # nengo.Connection(self.dec_am.cleaned_output_utilities,
        #                  dec_am2.dec_am_utils)
        # nengo.Connection(self.item_dcconv.output, dec_am2.input, synapse=0.01)

        # util_diff = cfg.make_thresh_ens_net(cfg.dec_am_min_diff)
        # nengo.Connection(self.dec_am.utilities, util_diff.input,
        #                  transform=[[1] * len(item_vocab.keys)], synapse=0.01)
        # nengo.Connection(dec_am2.utilities, util_diff.input,
        #                  transform=[[-1] * len(item_vocab.keys)], synapse=0.01)

        # util_diff_neg = cfg.make_thresh_ens_net(1 - cfg.dec_am_min_diff)
        # nengo.Connection(bias_node, util_diff_neg.input)
        # nengo.Connection(util_diff.output, util_diff_neg.input, transform=-2,
        #                  synapse=0.01)
        # nengo.Connection(self.dec_am.inhibit, util_diff_neg.input,
        #                  transform=-2)

        # util_diff_thresh = cfg.make_thresh_ens_net()   # Clean util_diff signal
        # nengo.Connection(bias_node, util_diff_thresh.input)
        # nengo.Connection(util_diff_neg.output, util_diff_thresh.input,
        #                  transform=-2, synapse=0.01)

        # Inhibit decoding associative memory when task != (DEC or DECW)
        # self.dec_am_task_inhibit = cfg.make_thresh_ens_net()
        # nengo.Connection(bias_node, self.dec_am_task_inhibit.input)
        # nengo.Connection(self.dec_am_task_inhibit.output,
        #                  self.dec_am.inhibit, synapse=0.01)
        # nengo.Connection(self.dec_am_task_inhibit.output,
        #                  dec_am2.inhibit, synapse=0.01)

        # Transform from visual WM to motor semantic pointer [for copy drawing
        # task]
        # from ..vision.lif_vision import am_threshold, am_vis_sps
        # ## self.vis_transform = \
        # ##    cfg.make_assoc_mem(am_vis_sps[:len(mtr_vocab.keys), :],
        # ##                       mtr_vocab.vectors, threshold=am_threshold,
        # ##                       inhibitable=True, inhibit_scale=3,
        # ##                       label='VISTRFM')
        # vis_tfrm_relay = cfg.make_ens_array(n_ensembles=cfg.mtr_dim,
        #                                     radius=mtr_sp_scale_factor)

        # Decoding output selector (selects between decoded from item WM or
        # transformed from visual WM)
        # self.select_out = nengo.Ensemble(cfg.n_neurons_ens, 1)
        # inhibit_am = cfg.make_thresh_ens_net()
        # inhibit_vis = cfg.make_thresh_ens_net()
        # nengo.Connection(self.select_out, inhibit_am.input)
        # nengo.Connection(self.select_out, inhibit_vis.input,
        #                  function=lambda x: 1 - x)
        # for ens in am_gated_ens.ensembles:
        #     nengo.Connection(inhibit_am.output, ens.neurons,
        #                      transform=[[-3]] * ens.n_neurons)
        # # ## nengo.Connection(inhibit_vis.output, self.vis_transform.inhibit)
        # for ens in vis_tfrm_relay.ensembles:
        #     nengo.Connection(inhibit_vis.output, ens.neurons,
        #                      transform=[[-3]] * ens.n_neurons)

        # # Decoding POS mem block gate signal generation (from motor system)
        # self.pos_mb_gate_bias = cfg.make_thresh_ens_net(n_neurons=100)
        # self.pos_mb_gate_sig = cfg.make_thresh_ens_net(0.3)

        # # Suppress pos_mb gate bias unless task=DEC + dec=FWD|REV|DECI|DECW
        # nengo.Connection(bias_node, self.pos_mb_gate_bias.input, transform=-1)

        # Use pos mb gate signal as inhibition for dec_am2 assoc mem
        # nengo.Connection(self.pos_mb_gate_sig, dec_am2.inhibit,
        #                  synapse=0.01)

        # self.dec_am_inhibit = cfg.make_thresh_ens_net(0.1)

        # Delay inhibition to dec_am and vis_transform to allow self.recall_mb
        # to capture data
        # dec_am_inhibit_delay = cfg.make_thresh_ens_net()
        # nengo.Connection(self.dec_am_inhibit.output,
        #                  dec_am_inhibit_delay.input, synapse=0.01)
        # nengo.Connection(dec_am_inhibit_delay.output, self.dec_am.inhibit,
        #                  synapse=0.01)
        # nengo.Connection(dec_am_inhibit_delay.output, dec_am2.inhibit,
        #                  synapse=0.01)
        # nengo.Connection(dec_am_inhibit_delay.output,
        #                  serial_decode.inhibit, synapse=0.01)

        # ## nengo.Connection(dec_am_inhibit_delay.output,
        # ##                 self.vis_transform.inhibit, synapse=0.01)
        # for ens in vis_tfrm_relay.ensembles:
        #     nengo.Connection(dec_am_inhibit_delay.output, ens.neurons,
        #                      transform=[[-3]] * ens.n_neurons, synapse=0.01)

        # Free recall decoding system
        # self.dec_am.add_output('item_output', item_vocab.vectors)
        # self.recall_mb = cfg.make_mem_block(vocab=vocab, gate_mode=1,
        #                                     reset_key=0)
        # self.recall_mb = cfg.make_mem_block(vocab=vocab, reset_key=0)
        # nengo.Connection(self.recall_mb.output, self.recall_mb.input,
        #                  transform=cfg.enc_mb_acc_fdbk_scale)
        # nengo.Connection(self.item_dcconv.B, self.recall_mb.input,
        #                  synapse=None)
        # nengo.Connection(self.dec_am_task_inhibit.output, self.recall_mb.reset)
        # nengo.Connection(self.pos_mb_gate_bias.output, self.recall_mb.gate,
        #                  transform=2, synapse=0.08)
        # nengo.Connection(self.pos_mb_gate_sig.output, self.recall_mb.gate,
        #                  transform=-2, synapse=0.01)

        # self.fr_dcconv = cfg.make_cir_conv(invert_b=True,
        #                                    input_magnitude=cfg.dcconv_radius)
        # nengo.Connection(self.item_dcconv.A, self.fr_dcconv.A, synapse=None)
        # nengo.Connection(self.recall_mb.output, self.fr_dcconv.B, transform=-1)

        # self.dec_am_fr = \
        #     cfg.make_assoc_mem(item_vocab.vectors, item_vocab.vectors,
        #                        inhibitable=True, inhibit_scale=5,
        #                        threshold=cfg.dec_fr_min_thresh,
        #                        default_output_vector=np.zeros(cfg.sp_dim))
        # nengo.Connection(self.fr_dcconv.output, self.dec_am_fr.input,
        #                  synapse=0.01)
        # nengo.Connection(self.dec_am_task_inhibit.output,
        #                  self.dec_am_fr.inhibit, synapse=0.01)
        # nengo.Connection(self.pos_mb_gate_sig.output, self.dec_am_fr.inhibit,
        #                  synapse=0.01)

        # Add output of free recall am as a small bias to dec_am
        # nengo.Connection(self.dec_am_fr.output, self.dec_am.input,
        #                  transform=cfg.dec_am_min_diff)

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
        # dec_am_y = self.dec_am.cleaned_output_utilities
        # dec_am_n = self.dec_am.output_default_ens

        # dec_am_diff_y = util_diff_thresh.output

        # fr_am_n = self.dec_am_fr.output_default_ens

        # output_know = cfg.make_thresh_ens_net(0.55)
        # output_unk = cfg.make_thresh_ens_net(0.80)
        # output_stop = cfg.make_thresh_ens_net(0.75)

        # nengo.Connection(dec_am_y, output_know.input,
        #                  transform=[[0.5] * self.dec_am.num_items],
        #                  synapse=0.01)
        # nengo.Connection(dec_am_diff_y, output_know.input, transform=0.5,
        #                  synapse=0.01)

        # nengo.Connection(bias_node, output_unk.input)
        # nengo.Connection(output_know.output, output_unk.input, transform=-5,
        #                  synapse=0.03)
        # nengo.Connection(self.pos_mb_gate_sig.output, output_unk.input,
        #                  transform=-2, synapse=0.03)
        # nengo.Connection(self.dec_am.inhibit, output_unk.input, transform=-2,
        #                  synapse=0.03)

        # nengo.Connection(dec_am_n, output_stop.input, transform=0.5,
        #                  synapse=0.03)
        # nengo.Connection(fr_am_n, output_stop.input, transform=0.5,
        #                  synapse=0.03)
        # nengo.Connection(self.pos_mb_gate_sig.output, output_stop.input,
        #                  transform=-2, synapse=0.03)

        # Output "UNK" vector to motor system.
        # Inhibited if output is KNOW, or if not using dec_am outputs
        # (i.e. during copy drawing)
        # bias_unk_vec = nengo.Ensemble(cfg.n_neurons_ens, 1)
        # nengo.Connection(bias_node, bias_unk_vec, synapse=None)
        # nengo.Connection(self.dec_am.inhibit, bias_unk_vec.neurons,
        #                  transform=[[-3]] * bias_unk_vec.n_neurons,
        #                  synapse=0.01)
        # nengo.Connection(self.pos_mb_gate_sig.output, bias_unk_vec.neurons,
        #                  transform=[[-3]] * bias_unk_vec.n_neurons,
        #                  synapse=0.01)
        # nengo.Connection(output_know.output, bias_unk_vec.neurons,
        #                  transform=[[-3]] * bias_unk_vec.n_neurons,
        #                  synapse=0.01)
        # nengo.Connection(output_stop.output, bias_unk_vec.neurons,
        #                  transform=[[-3]] * bias_unk_vec.n_neurons,
        #                  synapse=0.01)

        # Inhibit dec_am output (through am_gated_ens) if output is DON'T KNOW
        # for e in am_gated_ens.ensembles:
        #     nengo.Connection(output_unk.output, e.neurons,
        #                      transform=[[-3]] * e.n_neurons)

        # ############################ DEBUG ##################################
        self.item_dcconv = serial_decode.item_dcconv.output
        self.pos_recall_mb = free_recall_decode.pos_recall_mb.output
        self.pos_acc_input = free_recall_decode.pos_acc_input

        self.select_am = self.select_out.sel0
        self.select_vis = self.select_out.sel1

        self.am_out = nengo.Node(size_in=cfg.mtr_dim)
        self.vt_out = nengo.Node(size_in=cfg.mtr_dim)
        # nengo.Connection(self.dec_am.output, self.am_out, synapse=None)
        # nengo.Connection(self.vis_transform.output, self.vt_out, synapse=None) ## # noqa
        # nengo.Connection(vis_tfrm_relay.output, self.vt_out, synapse=None)

        self.am_utils = serial_decode.dec_am1.output_utilities
        self.am2_utils = serial_decode.dec_am2.output_utilities
        self.fr_utils = free_recall_decode.fr_am.output_utilities
        # self.util_diff = util_diff

        self.am_th_utils = serial_decode.dec_am1.cleaned_output_utilities
        self.fr_th_utils = free_recall_decode.fr_am.cleaned_output_utilities
        self.am_def_th_utils = serial_decode.dec_am1.output_default_ens
        self.fr_def_th_utils = free_recall_decode.fr_am.output_default_ens # noqa

        self.debug_task = nengo.Node(size_in=1)

        self.output_know = output_classify.output_know
        self.output_unk = output_classify.output_unk

        # self.util_diff_neg = util_diff_neg.output

        # ########################## END DEBUG ################################

        # Define network inputs and outputs
        self.dec_input = self.items_input
        self.pos_input = self.pos_input
        self.pos_acc_input = free_recall_decode.pos_acc_input
        self.vis_trfm_input = vis_trfm_decode.input

        # self.dec_output = nengo.Node(size_in=cfg.mtr_dim)
        # nengo.Connection(am_gated_ens.output, self.dec_output)
        # nengo.Connection(self.vis_transform.output, self.dec_output) ##
        # nengo.Connection(vis_tfrm_relay.output, self.dec_output)
        # nengo.Connection(bias_unk_vec, self.dec_output,
        #                  transform=np.matrix(mtr_unk_vocab['UNK'].v).T)

        self.dec_output = self.select_out.output
        self.output_stop = output_classify.output_stop

        self.dec_ind_output = nengo.Node(size_in=len(mtr_vocab.keys) + 1)
        nengo.Connection(serial_decode.dec_am1.cleaned_output_utilities,
                         self.dec_ind_output[:len(mtr_vocab.keys)],
                         synapse=None)
        nengo.Connection(output_classify.output_unk,
                         self.dec_ind_output[len(mtr_vocab.keys)],
                         synapse=None)

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            vis_am_utils = p_net.vis.am_utilities
            nengo.Connection(vis_am_utils[pos_mb_rst_sp_inds],
                             self.free_recall_decode.reset,
                             transform=[[cfg.mb_gate_scale] *
                                        len(pos_mb_rst_sp_inds)])
            # ##
        else:
            warn("InfoEncoding Module - Cannot connect from 'vis'")

        # Set up connections from vision module
        # Set up connections from vision module
        # if hasattr(p_net, 'vis'):
        #     # Only create this connection if we are using the LIF vision system
        #     if p_net.vis.mb_output.size_out == cfg.vis_dim:
        #         nengo.Connection(p_net.vis.mb_output, self.vis_trfm_input)
        # else:
        #     warn("TransformationSystem Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            # nengo.Connection(p_net.ps.dec, self.select_out,
            #                  transform=[dec_out_sel_sp_vecs * 1.0]) ##
            nengo.Connection(p_net.ps.dec, self.pos_mb_gate_bias.input,
                             transform=[dec_pos_gate_dec_sp_vecs * 1.0])
            nengo.Connection(p_net.ps.task, self.pos_mb_gate_bias.input,
                             transform=[dec_pos_gate_task_sp_vecs * 1.0])
            nengo.Connection(p_net.ps.task, self.dec_am_task_inhibit.input,
                             transform=[dec_pos_gate_task_sp_vecs * -1.0])

            # ###### DEBUG ########
            nengo.Connection(p_net.ps.dec, self.debug_task,
                             transform=[dec_pos_gate_dec_sp_vecs * 1.0])
        else:
            warn("InfoDecoding Module - Could not connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.pos_output, self.pos_input)
            nengo.Connection(p_net.enc.pos_acc_output, self.pos_acc_input)
        else:
            warn("InfoDecoding Module - Could not connect from 'enc'")

        # Set up connections from transform module
        # if hasattr(p_net, 'trfm'):
        #     nengo.Connection(p_net.trfm.output, self.dec_input,
        #                      transform=cfg.dcconv_item_in_scale)
        #     nengo.Connection(p_net.trfm.vis_trfm_output, self.vis_trfm_input)
        # else:
        #     warn("InfoDecoding Module - Could not connect from 'trfm'")

        # Set up connections from motor module
        if hasattr(p_net, 'mtr'):
            nengo.Connection(p_net.mtr.ramp_reset_hold.output,
                             self.pos_mb_gate_sig.input,
                             synapse=0.005, transform=5)
            nengo.Connection(p_net.mtr.ramp_reset_hold.output,
                             self.pos_mb_gate_sig.input,
                             synapse=0.08, transform=-10)

            nengo.Connection(p_net.mtr.ramp_reset_hold.output,
                             self.dec_am_inhibit.input,
                             synapse=0.005, transform=5)
            nengo.Connection(p_net.mtr.ramp_reset_hold.output,
                             self.dec_am_inhibit.input,
                             synapse=0.01, transform=-10)

            # nengo.Connection(p_net.mtr.ramp_50_75.output,
            #                  self.recall_mb.gate)
        else:
            warn("InfoDecoding Module - Could not connect from 'mtr'")
