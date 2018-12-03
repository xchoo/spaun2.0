import os
import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from .._networks import DetectChange

from .decoding import Serial_Recall_Network, Free_Recall_Network
from .decoding import Visual_Transform_Network, Output_Classification_Network
from .vision import vis_data
from .motor import mtr_data


class InfoDecoding(Module):
    def __init__(self, label="Information Dec", seed=None,
                 add_to_container=None):
        super(InfoDecoding, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(output=1)

        # ---------------------- Inputs and outputs ------------------------- #
        self.items_input = nengo.Node(size_in=vocab.sp_dim)
        self.pos_input = nengo.Node(size_in=vocab.sp_dim)

        # ----------------- Items input gated integrator -------------------- #
        # Holds item (list) vector stable if necessary
        items_input_mem = cfg.make_memory()
        nengo.Connection(self.items_input, items_input_mem.input,
                         synapse=None)

        # ----------------- Inhibition signal generation -------------------- #
        # Inhibition signal for when TASK != DEC
        self.dec_am_task_inhibit = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, self.dec_am_task_inhibit.input,
                         synapse=None)

        # Generic inhibition signal?
        self.dec_am_inhibit = cfg.make_thresh_ens_net(0.1)

        # Inhibition signal when position vector changes
        self.pos_change = DetectChange(dimensions=vocab.sp_dim,
                                       n_neurons=cfg.n_neurons_ens)
        nengo.Connection(self.pos_input, self.pos_change.input)

        # ---------- Decoding POS mem block gate signal generation ---------- #
        # Decoding POS mem block gate signal generation (from motor system)
        self.pos_mb_gate_bias = cfg.make_thresh_ens_net(n_neurons=200)
        self.pos_mb_gate_sig = cfg.make_thresh_ens_net(0.3)

        # Bias does ...?
        # Gate signal does ...?

        # Suppress pos_mb gate bias unless task=DEC+L + dec=FWD|REV|DECI|DECW
        nengo.Connection(bias_node, self.pos_mb_gate_bias.input, transform=-1)

        # -------------------- Serial decoding network ---------------------- #
        serial_decode = Serial_Recall_Network(vocab.item, vocab.mtr)
        nengo.Connection(items_input_mem.output, serial_decode.items_input,
                         transform=cfg.dcconv_item_in_scale, synapse=None)
        nengo.Connection(self.pos_input, serial_decode.pos_input,
                         synapse=None)

        # Inhibitory connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         serial_decode.inhibit, synapse=0.01)
        nengo.Connection(self.pos_change.output, serial_decode.inhibit)

        # ---------------- Free recall decoding network --------------------- #
        self.free_recall_decode = Free_Recall_Network(vocab.item, vocab.pos,
                                                      vocab.mtr)
        nengo.Connection(items_input_mem.output,
                         self.free_recall_decode.items_input,
                         transform=cfg.dec_fr_item_in_scale, synapse=None)
        nengo.Connection(self.pos_input, self.free_recall_decode.pos_input,
                         synapse=None)

        # Add output of free recall am as a small bias to dec_am
        nengo.Connection(self.free_recall_decode.output,
                         serial_decode.items_input,
                         transform=cfg.dec_fr_to_am_scale)

        # Gating connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         self.free_recall_decode.reset)
        nengo.Connection(self.pos_mb_gate_bias.output,
                         self.free_recall_decode.gate, transform=2,
                         synapse=0.03)
        # - Large synapse here to smooth out bias signal
        #   (it's kinda wibbly-wobbly)
        nengo.Connection(self.pos_mb_gate_sig.output,
                         self.free_recall_decode.gate, transform=-4,
                         synapse=0.01)

        # Inhibitory connections
        nengo.Connection(self.dec_am_task_inhibit.output,
                         self.free_recall_decode.inhibit, synapse=0.01)
        nengo.Connection(self.pos_change.output,
                         self.free_recall_decode.inhibit)

        # ------------- Visual transform decoding network ------------------- #
        if vocab.vis_dim > 0:
            # Takes visual SP and transforms them to the 'copy-drawn' motor SP
            # TODO: Use output of Serial Decode instead of replicating code?
            # copy_draw_tfrm_data = \
            #     np.load(os.path.join(mtr_data.filepath, cfg.mtr_module,
            #                          '_'.join([cfg.vis_module,
            #                                    'copydraw_trfms.npz'])),
            #             encoding='latin1')
            # copy_draw_trfms_x = copy_draw_tfrm_data['trfms_x']
            # copy_draw_trfms_y = copy_draw_tfrm_data['trfms_y']

            copy_draw_trfms_x = np.random.randn(10, 200, 25)
            copy_draw_trfms_y = np.random.randn(10, 200, 25)

            vis_trfm_decode = \
                Visual_Transform_Network(vocab.vis,
                                         vis_data.am_threshold * 0.5, 1,
                                         copy_draw_trfms_x,
                                         copy_draw_trfms_y, vocab.mtr,
                                         mtr_data.sp_scaling_factor)

            # Inhibitory connections
            nengo.Connection(self.dec_am_task_inhibit.output,
                             vis_trfm_decode.am_inhibit, synapse=0.01)
        else:
            from .decoding.vis_trfm_net import Dummy_Visual_Transform_Network
            vis_trfm_decode = \
                Dummy_Visual_Transform_Network(vectors_in=vocab.item.vectors,
                                               vectors_out=vocab.mtr.vectors)

        # -------- Output classification (know / unknown / stop) system ----- #
        self.output_classify = Output_Classification_Network()
        nengo.Connection(serial_decode.dec_success,
                         self.output_classify.sr_utils_y,
                         transform=[[1.0] * serial_decode.dec_am1.n_items],
                         synapse=0.01)
        nengo.Connection(serial_decode.am_utils_diff,
                         self.output_classify.sr_utils_diff, synapse=0.01)
        nengo.Connection(serial_decode.dec_failure,
                         self.output_classify.sr_utils_n, synapse=0.03)
        nengo.Connection(self.free_recall_decode.dec_failure,
                         self.output_classify.fr_utils_n, synapse=0.03)

        # Output classification inhibitory signals
        # - Inhibit UNK when am's are inhibited.
        # - Inhibit UNK and STOP when pos_gate_sig is HIGH
        #   (i.e. decoding system is doing things)
        nengo.Connection(self.dec_am_inhibit.output,
                         self.output_classify.output_unk_inhibit, synapse=0.03)
        nengo.Connection(self.pos_mb_gate_sig.output,
                         self.output_classify.output_unk_inhibit, synapse=0.03)
        nengo.Connection(self.pos_mb_gate_sig.output,
                         self.output_classify.output_stop_inhibit,
                         synapse=0.03)

        # ----------------------- Output selector --------------------------- #
        # 0: Serial decoder output
        # 1: Copy drawing transform output
        # 2: Free recall decoder output
        # 3: UNK mtr SP output
        # 4: NULL (all zeros) output
        self.select_out = cfg.make_selector(5,
                                            radius=mtr_data.sp_scaling_factor,
                                            dimensions=vocab.mtr_dim,
                                            make_ens_func=cfg.make_ens_array,
                                            ens_dimensions=1,
                                            threshold_sel_in=True)

        # Connections for sel0 - SR
        nengo.Connection(serial_decode.output, self.select_out.input0)
        nengo.Connection(self.output_classify.output_unk,
                         self.select_out.sel0, transform=-1)
        nengo.Connection(self.output_classify.output_stop,
                         self.select_out.sel4, transform=-1)

        # Connections for sel1 - Copy Drawing
        nengo.Connection(vis_trfm_decode.output, self.select_out.input1)
        nengo.Connection(self.output_classify.output_unk,
                         self.select_out.sel1, transform=-1)
        nengo.Connection(self.output_classify.output_stop,
                         self.select_out.sel4, transform=-1)

        # Connections for sel2 - FR
        nengo.Connection(self.free_recall_decode.mtr_output,
                         self.select_out.input2)
        nengo.Connection(self.output_classify.output_unk,
                         self.select_out.sel2, transform=-1)
        nengo.Connection(self.output_classify.output_stop,
                         self.select_out.sel4, transform=-1)

        # Connections for sel3 - UNK
        nengo.Connection(bias_node, self.select_out.input3,
                         transform=vocab.mtr_unk['UNK'].v[:, None])
        nengo.Connection(self.output_classify.output_unk,
                         self.select_out.sel3)
        nengo.Connection(self.output_classify.output_stop,
                         self.select_out.sel4, transform=-1)

        # Connections for sel4 - NULL
        nengo.Connection(self.output_classify.output_stop,
                         self.select_out.sel4)

        # ############################ DEBUG ##################################
        self.item_dcconv = serial_decode.item_dcconv.output
        self.pos_recall_mb = self.free_recall_decode.pos_recall_mb.output
        self.pos_recall_mb_in = self.free_recall_decode.pos_recall_mb.input
        self.pos_recall_mb_gate = self.free_recall_decode.pos_recall_mb.gate
        self.pos_acc_input = self.free_recall_decode.pos_acc_input
        self.fr_recall_mb = self.free_recall_decode.pos_recall_mb
        self.fr_dcconv = self.free_recall_decode.fr_dcconv
        self.fr_am_out = self.free_recall_decode.fr_am.output

        self.select_am = self.select_out.sel0
        self.select_vis = self.select_out.sel1

        self.am_out = nengo.Node(size_in=vocab.mtr_dim)
        self.vt_out = nengo.Node(size_in=vocab.mtr_dim)
        # nengo.Connection(self.dec_am.output, self.am_out, synapse=None)
        # nengo.Connection(self.vis_transform.output, self.vt_out, synapse=None) ## # noqa
        # nengo.Connection(vis_tfrm_relay.output, self.vt_out, synapse=None)

        self.am_utils = serial_decode.dec_am1.linear_output
        self.am2_utils = serial_decode.dec_am2.linear_output
        self.fr_utils = self.free_recall_decode.fr_am.output_utilities
        self.util_diff = serial_decode.am_utils_diff

        self.am_th_utils = serial_decode.dec_am1.cleaned_output_utilities
        self.fr_th_utils = self.free_recall_decode.fr_am.cleaned_output_utilities  # noqa
        self.am_def_th_utils = serial_decode.dec_am1.output_default_ens
        self.fr_def_th_utils = self.free_recall_decode.fr_am.output_default_ens # noqa

        self.out_class_sr_y = self.output_classify.sr_utils_y
        self.out_class_sr_diff = self.output_classify.sr_utils_diff
        self.out_class_sr_n = self.output_classify.sr_utils_n

        self.debug_task = nengo.Node(size_in=1)

        self.output_know = self.output_classify.output_know
        self.output_unk = self.output_classify.output_unk

        self.item_dcconv_a = serial_decode.item_dcconv.A
        self.item_dcconv_b = serial_decode.item_dcconv.B

        self.serial_decode = serial_decode

        # self.util_diff_neg = util_diff_neg.output
        self.sel_signals = nengo.Node(size_in=5)
        for n in range(5):
            nengo.Connection(getattr(self.select_out, 'sel%d' % n),
                             self.sel_signals[n], synapse=None)

        self.vis_trfm_decode = vis_trfm_decode

        # ########################## END DEBUG ################################

        # Define network inputs and outputs
        self.items_input = self.items_input
        self.items_input_mem = items_input_mem
        self.pos_input = self.pos_input
        self.pos_acc_input = self.free_recall_decode.pos_acc_input
        self.vis_trfm_input = vis_trfm_decode.input

        self.output = self.select_out.output
        self.output_stop = self.output_classify.output_stop

        # Direct motor (digit) index output to the experimenter system
        self.dec_ind_output = nengo.Node(size_in=len(vocab.mtr.keys) + 1)
        nengo.Connection(serial_decode.dec_am1.cleaned_output_utilities,
                         self.dec_ind_output[:len(vocab.mtr.keys)],
                         synapse=None)
        nengo.Connection(self.output_classify.output_unk,
                         self.dec_ind_output[len(vocab.mtr.keys)],
                         synapse=None)

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(parent_net, 'vis'):
            fr_pos_mb_rst_sp_vecs = vocab.main.parse('A+QM').v
            nengo.Connection(parent_net.vis.output,
                             self.free_recall_decode.reset,
                             transform=[fr_pos_mb_rst_sp_vecs])

            nengo.Connection(p_net.vis.mb_output, self.vis_trfm_input)
        else:
            warn("InfoEncoding Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            # Connections for sel0 - SR
            dec_out_sr_sp_vecs = vocab.main.parse('FWD+REV+CNT+DECI').v
            nengo.Connection(p_net.ps.dec, self.select_out.sel0,
                             transform=[dec_out_sr_sp_vecs])

            # Connections for sel1 - Copy Drawing
            dec_out_copy_draw_sp_vecs = vocab.main.parse('DECW').v
            nengo.Connection(p_net.ps.dec, self.select_out.sel1,
                             transform=[dec_out_copy_draw_sp_vecs])

            # Connections for sel2 - FR
            dec_out_fr_sp_vecs = vocab.main.parse('0').v  # TODO: Implement
            nengo.Connection(p_net.ps.dec, self.select_out.sel2,
                             transform=[dec_out_fr_sp_vecs])

            # Connections for gate signals
            dec_pos_gate_dec_sp_vecs = vocab.main.parse('DECW+DECI+FWD+REV').v
            nengo.Connection(p_net.ps.dec, self.pos_mb_gate_bias.input,
                             transform=[dec_pos_gate_dec_sp_vecs],
                             synapse=0.02)

            dec_pos_gate_task_sp_vecs = vocab.main.parse('DEC').v
            nengo.Connection(p_net.ps.task, self.pos_mb_gate_bias.input,
                             transform=[dec_pos_gate_task_sp_vecs],
                             synapse=0.02)

            # Connections for inhibitory signals
            nengo.Connection(p_net.ps.task, self.dec_am_task_inhibit.input,
                             transform=[dec_pos_gate_task_sp_vecs * -1.5])

            # Inhibit FR for induction, learning, counting, react, and instr
            # tasks
            # - also set fr_utils_n to high
            dec_inhibit_fr_sp_vecs = vocab.main.parse('DECI+L+C+REACT+INSTR').v
            nengo.Connection(p_net.ps.task, self.free_recall_decode.inhibit,
                             transform=[dec_inhibit_fr_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.free_recall_decode.inhibit,
                             transform=[dec_inhibit_fr_sp_vecs])

            nengo.Connection(p_net.ps.task, self.output_classify.fr_utils_n,
                             transform=[dec_inhibit_fr_sp_vecs])
            nengo.Connection(p_net.ps.dec, self.output_classify.fr_utils_n,
                             transform=[dec_inhibit_fr_sp_vecs])

            # Inhibit output stop during counting task
            dec_inhibit_output_stop_sp_vecs = vocab.main.parse('CNT').v
            nengo.Connection(p_net.ps.dec,
                             self.output_classify.output_stop_inhibit,
                             transform=[dec_inhibit_output_stop_sp_vecs])

            # ###### DEBUG ########
            dec_pos_gate_dec_sp_vecs = vocab.main.parse('DECW+DECI+FWD+REV').v
            nengo.Connection(p_net.ps.dec, self.debug_task,
                             transform=[dec_pos_gate_dec_sp_vecs])
        else:
            warn("InfoDecoding Module - Could not connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.pos_output, self.pos_input)
            nengo.Connection(p_net.enc.pos_acc_output, self.pos_acc_input)
        else:
            warn("InfoDecoding Module - Could not connect from 'enc'")

        # Set up connections from transform module
        if hasattr(p_net, 'trfm'):
            nengo.Connection(p_net.trfm.output, self.items_input)
        else:
            warn("InfoDecoding Module - Could not connect from 'trfm'")

        # Set up connections from motor module
        if hasattr(p_net, 'mtr'):
            # nengo.Connection(p_net.mtr.ramp_reset_hold,
            #                  self.pos_mb_gate_sig.input,
            #                  synapse=0.005, transform=5)
            # nengo.Connection(p_net.mtr.ramp_reset_hold,
            #                  self.pos_mb_gate_sig.input,
            #                  synapse=0.08, transform=-10)
            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.pos_mb_gate_sig.input,
                             synapse=0.005)

            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.dec_am_inhibit.input,
                             synapse=0.005, transform=5)
            nengo.Connection(p_net.mtr.ramp_reset_hold,
                             self.dec_am_inhibit.input,
                             synapse=0.01, transform=-10)

            # ----------- Connection to items input memory --------------------
            # Gate memory when pen is down (to stop output representation
            # from changing when writing a digit)
            nengo.Connection(p_net.mtr.pen_down, self.items_input_mem.gate)
        else:
            warn("InfoDecoding Module - Could not connect from 'mtr'")
