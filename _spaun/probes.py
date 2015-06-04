import os
import numpy as np

import nengo

from .config import cfg
from .vocabs import vis_vocab, pos_vocab, enum_vocab # noqa
from .vocabs import ps_task_vocab, ps_state_vocab, ps_dec_vocab, mtr_vocab  # noqa
from .vocabs import mtr_disp_vocab, item_vocab, pos1_vocab, vocab  # noqa
from .modules.working_memory import WorkingMemoryDummy
from .modules.transform_system import TransformationSystemDummy


def idstr(p):
    if not isinstance(p, nengo.Probe):
        return str(p)
    else:
        return str(id(p))


def gen_probe_list(probes, probes_to_legend=[]):
    probe_list = map(idstr, probes)

    if not isinstance(probes_to_legend, list):
        probes_to_legend = [probes_to_legend]

    for probe in probes_to_legend:
        probe_list[probe_list.index(idstr(probe))] += '*'

    if probe_list[-1] != '0':
        probe_list.append('0')

    return probe_list


def config_and_setup_probes(model):
    version = 3.0

    config_filename = cfg.probe_data_filename[:-4] + '_cfg.npz'

    probe_list, vocab_dict = setup_probes(model)
    config_data = {'sp_dim': cfg.sp_dim, 'probe_list': probe_list,
                   'vocab_dict': vocab_dict, 'prim_vocab': vocab,
                   'dt': cfg.sim_dt, 'version': version}

    np.savez_compressed(os.path.join(cfg.data_dir, config_filename),
                        **config_data)


def setup_probes(model):
    # return setup_probes1(model)
    # return setup_probes2(model)
    # return setup_probes3(model)
    # return setup_probes4(model)
    # return setup_probes_vis(model)
    return setup_probes_generic(model)


def setup_probes_vis(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3])
    probe_list[1] += '*'
    probe_list[3] += '*'

    vocab_dict = {idstr(pvs1): vis_vocab}

    return (probe_list, vocab_dict)


def setup_probes1(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)
        # pvs4 = nengo.Probe(model.vis.mb_output, synapse=0.005)
        # pvs5 = nengo.Probe(model.vis.vis_mb.gate, synapse=0.005)
        # pvs6 = nengo.Probe(model.vis.vis_net.output, synapse=0.005)

        pps1 = nengo.Probe(model.ps.task, synapse=0.005)
        # pps2 = nengo.Probe(model.ps.ps_task_mb.gate, synapse=0.005)
        # pps3 = nengo.Probe(model.ps.ps_task_mb.gateX, synapse=0.005)
        # pps4 = nengo.Probe(model.ps.ps_task_mb.gateN, synapse=0.005)
        # pps5 = nengo.Probe(model.ps.ps_task_mb.mem1.output, synapse=0.005)

        # pth1 = nengo.Probe(model.thal.output, synapse=0.005)

        pen1 = nengo.Probe(model.enc.pos_mb.gate, synapse=0.005)
        pen2 = nengo.Probe(model.enc.pos_mb.gateX, synapse=0.005)
        pen3 = nengo.Probe(model.enc.pos_mb.gateN, synapse=0.005)
        pen4 = nengo.Probe(model.enc.pos_mb.mem2.output, synapse=0.005)
        pen5 = nengo.Probe(model.enc.pos_mb.mem1.output, synapse=0.005)
        # pen6 = nengo.Probe(model.enc.pos_mb.am.output, synapse=0.005)
        pen6 = nengo.Probe(model.enc.pos_mb.reset, synapse=0.005)

        pmm1 = nengo.Probe(model.mem.output, synapse=0.005)
        pmm2 = nengo.Probe(model.mem.mb1a.gate, synapse=0.005)
        pmm3 = nengo.Probe(model.mem.mb1a.gateX, synapse=0.005)
        pmm4 = nengo.Probe(model.mem.mb1a.gateN, synapse=0.005)
        pmm5 = nengo.Probe(model.mem.output, synapse=0.005)
        # pmm5 = nengo.Probe(model.mem.mb1a.output, synapse=0.005)
        # pmm6 = nengo.Probe(model.mem.mb1b.output, synapse=0.005)
        # pmm7 = nengo.Probe(model.mem.mb1b.mem1.output, synapse=0.005)
        # pmm8 = nengo.Probe(model.mem.mb1b.mem2.output, synapse=0.005)
        # pmm9 = nengo.Probe(model.mem.mem_in, synapse=0.005)
        pmm6 = nengo.Probe(model.mem.mb1a.mem1.output, synapse=0.005)
        pmm7 = nengo.Probe(model.mem.mb1a.mem2.output, synapse=0.005)
        pmm8 = nengo.Probe(model.mem.mb1b.mem1.output, synapse=0.005)
        pmm9 = nengo.Probe(model.mem.mb1b.mem2.output, synapse=0.005)

        pde1 = nengo.Probe(model.dec.item_dcconv.output, synapse=0.005)
        pde2 = nengo.Probe(model.dec.select_am, synapse=0.005)
        pde3 = nengo.Probe(model.dec.select_vis, synapse=0.005)
        pde4 = nengo.Probe(model.dec.am_out, synapse=0.01)
        pde5 = nengo.Probe(model.dec.vt_out, synapse=0.005)
        pde6 = nengo.Probe(model.dec.pos_mb_gate_sig, synapse=0.005)
        pde7 = nengo.Probe(model.dec.util_diff_neg, synapse=0.005)
        pde8 = nengo.Probe(model.dec.am_utils, synapse=0.005)
        pde9 = nengo.Probe(model.dec.am2_utils, synapse=0.005)
        pde10 = nengo.Probe(model.dec.util_diff, synapse=0.005)
        pde11 = nengo.Probe(model.dec.recall_mb.output, synapse=0.005)
        pde12 = nengo.Probe(model.dec.dec_am_fr.output, synapse=0.005)
        pde13 = nengo.Probe(model.dec.dec_am.item_output, synapse=0.005)
        # pde14 = nengo.Probe(model.dec.recall_mb.mem1.output, synapse=0.005)
        pde15 = nengo.Probe(model.dec.output_know, synapse=0.005)
        pde16 = nengo.Probe(model.dec.output_unk, synapse=0.005)
        pde18 = nengo.Probe(model.dec.output_stop, synapse=0.005)
        pde19 = nengo.Probe(model.dec.am_th_utils, synapse=0.005)
        pde20 = nengo.Probe(model.dec.fr_th_utils, synapse=0.005)
        pde21 = nengo.Probe(model.dec.dec_output, synapse=0.005)
        pde22 = nengo.Probe(model.dec.dec_am_fr.input, synapse=0.005)
        pde23 = nengo.Probe(model.dec.am_def_th_utils, synapse=0.005)
        pde24 = nengo.Probe(model.dec.fr_def_th_utils, synapse=0.005)
        pde25 = nengo.Probe(model.dec.fr_utils, synapse=0.005)

        pmt1 = nengo.Probe(model.mtr.ramp, synapse=0.005)
        pmt2 = nengo.Probe(model.mtr.ramp_reset_hold, synapse=0.005)
        pmt3 = nengo.Probe(model.mtr.motor_stop_input, synapse=0.005)
        pmt4 = nengo.Probe(model.mtr.motor_init, synapse=0.005)
        pmt5 = nengo.Probe(model.mtr.motor_go, synapse=0.005)

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3, 0,
                             p0, pps1, pmm1, pmm6, pmm7, pmm8, pmm9, 0,
                             p0, pen1, pen2, pen3, pen4, pen5, pen6, 0,
                             p0, pde1, pde2, pde3, pde4, pde5, pde6, 0,
                             p0, pmt1, pmt2, pmt3, pmt4, pmt5, 0,
                             p0, pde8, pde9, pde10, pde7, 0,
                             p0, pmm5, pde11, pde12, pde13, pde22, 0,
                             p0, pde19, pde20, pde15, pde16, pde18, 0,
                             p0, pmt1, pde21, pde23, pde24, pde25])
    vocab_dict = {idstr(pvs1): vis_vocab,
                  idstr(pps1): ps_task_vocab,
                  idstr(pmm1): enum_vocab,
                  idstr(pmm6): enum_vocab,
                  idstr(pmm7): enum_vocab,
                  idstr(pmm8): enum_vocab,
                  idstr(pmm9): enum_vocab,
                  idstr(pmm5): item_vocab,
                  idstr(pen4): pos_vocab,
                  idstr(pen5): pos_vocab,
                  idstr(pde1): item_vocab,
                  idstr(pde4): mtr_disp_vocab,
                  idstr(pde5): mtr_disp_vocab,
                  idstr(pde11): item_vocab,
                  idstr(pde12): item_vocab,
                  idstr(pde13): item_vocab,
                  idstr(pde21): mtr_disp_vocab,
                  idstr(pde22): item_vocab}

    return (probe_list, vocab_dict)


def setup_probes2(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

        pps1 = nengo.Probe(model.ps.task, synapse=0.005)
        pps2 = nengo.Probe(model.ps.ps_task_mb.gate, synapse=0.005)
        pps3 = nengo.Probe(model.ps.ps_task_mb.reset, synapse=0.005)
        pps4 = nengo.Probe(model.ps.state, synapse=0.005)
        pps5 = nengo.Probe(model.ps.ps_state_mb.gate, synapse=0.005)
        pps6 = nengo.Probe(model.ps.ps_state_mb.reset, synapse=0.005)
        pps7 = nengo.Probe(model.ps.dec, synapse=0.005)
        pps8 = nengo.Probe(model.ps.ps_dec_mb.gate, synapse=0.005)
        pps9 = nengo.Probe(model.ps.ps_dec_mb.reset, synapse=0.005)

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3, 0,
                             p0, pps1, pps2, pps3, 0,
                             p0, pps4, pps5, pps6, 0,
                             p0, pps7, pps8, pps9])
    probe_list[1] += "*"
    probe_list[6] += "*"
    probe_list[11] += "*"
    probe_list[16] += "*"
    vocab_dict = {idstr(pvs1): vis_vocab,
                  idstr(pps1): ps_task_vocab,
                  idstr(pps4): ps_state_vocab,
                  idstr(pps7): ps_dec_vocab}

    return (probe_list, vocab_dict)


def setup_probes3(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

        pps1 = nengo.Probe(model.ps.task, synapse=0.005)
        pps2 = nengo.Probe(model.ps.ps_task_mb.gate, synapse=0.005)
        pps3 = nengo.Probe(model.ps.ps_task_mb.reset, synapse=0.005)
        pps4 = nengo.Probe(model.ps.state, synapse=0.005)
        pps5 = nengo.Probe(model.ps.ps_state_mb.gate, synapse=0.005)
        pps6 = nengo.Probe(model.ps.ps_state_mb.reset, synapse=0.005)
        pps7 = nengo.Probe(model.ps.dec, synapse=0.005)
        pps8 = nengo.Probe(model.ps.ps_dec_mb.gate, synapse=0.005)
        pps9 = nengo.Probe(model.ps.ps_dec_mb.reset, synapse=0.005)

        pps10 = nengo.Probe(model.ps.ps_task_mb.mem1.mem.thresh, synapse=0.005)
        pps11 = nengo.Probe(model.ps.ps_task_mb.mem2.mem.thresh, synapse=0.005)
        pps12 = nengo.Probe(model.ps.ps_task_mb.mem1.diff.output,
                            synapse=0.005)
        pps13 = nengo.Probe(model.ps.ps_task_mb.mem1.input, synapse=0.005)

        pbg1 = nengo.Probe(model.bg.input, synapse=0.005)
        pbg2 = nengo.Probe(model.bg.output, synapse=0.005)

        pmm1 = nengo.Probe(model.mem.mb1, synapse=0.005)
        pmm2 = nengo.Probe(model.mem.mb1_gate, synapse=0.005)
        pmm3 = nengo.Probe(model.mem.mb1_reset, synapse=0.005)
        pmm4 = nengo.Probe(model.mem.mb2, synapse=0.005)
        pmm5 = nengo.Probe(model.mem.mb2_gate, synapse=0.005)
        pmm6 = nengo.Probe(model.mem.mb2_reset, synapse=0.005)
        pmm7 = nengo.Probe(model.mem.mb3, synapse=0.005)
        pmm8 = nengo.Probe(model.mem.mb3_gate, synapse=0.005)
        pmm9 = nengo.Probe(model.mem.mb3_reset, synapse=0.005)

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3, 0,
                             p0, pps1, pps2, pps3, 0,
                             p0, pps4, pps5, pps6, 0,
                             p0, pps7, pps8, pps9, 0,
                             p0, pmm1, pmm2, pmm3, 0,
                             p0, pmm4, pmm5, pmm6, 0,
                             p0, pmm7, pmm8, pmm9, 0,
                             p0, pbg1, pbg2, 0,
                             p0, pps10, pps11, pps12, pps13])
    probe_list[1] += "*"
    probe_list[6] += "*"
    probe_list[11] += "*"
    probe_list[16] += "*"
    # probe_list[21] += "*"
    # probe_list[26] += "*"
    # probe_list[31] += "*"
    probe_list[36] += "*"
    probe_list[37] += "*"
    vocab_dict = {idstr(pvs1): vis_vocab,
                  idstr(pps1): ps_task_vocab,
                  idstr(pps4): ps_state_vocab,
                  idstr(pps7): ps_dec_vocab,
                  idstr(pmm1): enum_vocab,
                  idstr(pmm4): enum_vocab,
                  idstr(pmm7): enum_vocab,
                  idstr(pps13): ps_task_vocab}

    return (probe_list, vocab_dict)


def setup_probes_generic(model):
    with model:
        vocab_dict = {}
        probe_list = []

        sub_vocab1 = enum_vocab.create_subset(['POS1*ONE', 'POS1*TWO',
                                               'POS1*THR', 'POS1*FOR',
                                               'POS2*TWO', 'POS2*THR',
                                               'POS3*THR', 'POS3*FOR'])

        sub_vocab2 = vocab.create_subset(['ADD'])
        sub_vocab2.add('ADD*ADD', vocab.parse('ADD*ADD'))
        sub_vocab2.add('ADD*ADD*ADD', vocab.parse('ADD*ADD*ADD'))

        if hasattr(model, 'stim'):
            p0 = nengo.Probe(model.stim.output)
        else:
            p0 = 0

        if hasattr(model, 'vis') and True:
            pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
            pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
            pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

            # probes = gen_probe_list(['vis', p0, pvs1, pvs2, pvs3], pvs1)
            probes = gen_probe_list(['vis', p0, pvs1, pvs2, pvs3])
            probe_list.extend(probes)
            vocab_dict[idstr(pvs1)] = vis_vocab

        # ############ FOR DEBUGGING VIS DETECT SYSTEM ########################
        # if hasattr(model, 'vis') and True:
        #     pvsd1 = nengo.Probe(model.vis.detect_change_net.input_diff, synapse=0.005)
        #     pvsd2 = nengo.Probe(model.vis.detect_change_net.item_detect, synapse=0.005)
        #     pvsd3 = nengo.Probe(model.vis.detect_change_net.blank_detect, synapse=0.005)

        #     probes = gen_probe_list(['vis detect', p0, pvsd1, pvsd2, pvsd3])
        #     probe_list.extend(probes)

        if hasattr(model, 'ps') and True:
            pps1 = nengo.Probe(model.ps.task, synapse=0.005)
            pps2 = nengo.Probe(model.ps.state, synapse=0.005)
            pps3 = nengo.Probe(model.ps.dec, synapse=0.005)

            pps4 = nengo.Probe(model.ps.ps_task_mb.mem1.output, synapse=0.005)
            pps5 = nengo.Probe(model.ps.ps_task_mb.mem2.output, synapse=0.005)
            pps6 = nengo.Probe(model.ps.ps_task_mb.mem1.input, synapse=0.005)

            probes = gen_probe_list(['ps', p0, pps1, pps2, pps3, 0,
                                     'ps_task', p0, pps1, pps6, pps4, pps5],
                                    [pps1, pps2, pps3, pps4, pps5, pps6])
            probe_list.extend(probes)

            vocab_dict[idstr(pps1)] = ps_task_vocab
            vocab_dict[idstr(pps2)] = ps_state_vocab
            vocab_dict[idstr(pps3)] = ps_dec_vocab
            vocab_dict[idstr(pps4)] = ps_task_vocab
            vocab_dict[idstr(pps5)] = ps_task_vocab
            vocab_dict[idstr(pps6)] = ps_task_vocab

        if hasattr(model, 'enc') and True:
            pen1 = nengo.Probe(model.enc.pos_mb.gate, synapse=0.005)
            # pen2 = nengo.Probe(model.enc.pos_mb.gateX, synapse=0.005)
            # pen3 = nengo.Probe(model.enc.pos_mb.gateN, synapse=0.005)
            pen4 = nengo.Probe(model.enc.pos_mb.mem2.output, synapse=0.005)
            pen5 = nengo.Probe(model.enc.pos_mb.mem1.output, synapse=0.005)
            # pen6 = nengo.Probe(model.enc.pos_mb.am.output, synapse=0.005)
            pen6 = nengo.Probe(model.enc.pos_mb.reset, synapse=0.005)

            probes = gen_probe_list(['enc', p0, pen1, pen4, pen5, pen6],
                                    [pen4, pen5])
            probe_list.extend(probes)

            vocab_dict[idstr(pen4)] = pos_vocab
            vocab_dict[idstr(pen5)] = pos_vocab

        if hasattr(model, 'mem') and True:
            pmm1 = nengo.Probe(model.mem.mb1, synapse=0.005)
            pmm2 = nengo.Probe(model.mem.mb1_gate, synapse=0.005)
            pmm3 = nengo.Probe(model.mem.mb1_reset, synapse=0.005)
            pmm4 = nengo.Probe(model.mem.mb2, synapse=0.005)
            pmm5 = nengo.Probe(model.mem.mb2_gate, synapse=0.005)
            pmm6 = nengo.Probe(model.mem.mb2_reset, synapse=0.005)
            pmm7 = nengo.Probe(model.mem.mb3, synapse=0.005)
            pmm8 = nengo.Probe(model.mem.mb3_gate, synapse=0.005)
            pmm9 = nengo.Probe(model.mem.mb3_reset, synapse=0.005)

            probes = gen_probe_list(['mb1', p0, pmm1, pmm2, pmm3, 0,
                                     'mb2', p0, pmm4, pmm5, pmm6, 0,
                                     'mb3', p0, pmm7, pmm8, pmm9],
                                    [pmm1, pmm4, pmm7])
            probe_list.extend(probes)
            vocab_dict[idstr(pmm1)] = sub_vocab1
            vocab_dict[idstr(pmm4)] = sub_vocab1
            vocab_dict[idstr(pmm7)] = sub_vocab1

        if hasattr(model, 'mem') and True:
            pmm10 = nengo.Probe(model.mem.mbave_in, synapse=0.005)

            probes = gen_probe_list(['mbave', p0, pmm10], pmm10)
            probe_list.extend(probes)
            vocab_dict[idstr(pmm10)] = sub_vocab1

        if (hasattr(model, 'mem') and not isinstance(model.mem,
                                                     WorkingMemoryDummy)):
            pmm11 = nengo.Probe(model.mem.mb1a.gateX, synapse=0.005)
            pmm12 = nengo.Probe(model.mem.mb1a.gateN, synapse=0.005)
            pmm13 = nengo.Probe(model.mem.mb1a.mem1.gate, synapse=0.005)
            pmm14 = nengo.Probe(model.mem.mb1a.mem1.reset, synapse=0.005)

            probes = gen_probe_list(['mb1 sigs', p0, pmm11, pmm12, pmm13,
                                     pmm14])
            probe_list.extend(probes)

        if hasattr(model, 'trfm') and \
                not isinstance(model.trfm, TransformationSystemDummy):
            ptf1 = nengo.Probe(model.trfm.select_cc1a.output, synapse=0.005)
            ptf2 = nengo.Probe(model.trfm.select_cc1b.output, synapse=0.005)
            ptf3 = nengo.Probe(model.trfm.cconv1.output, synapse=0.005)
            ptf3b = nengo.Probe(model.trfm.cconv1.output, synapse=0.005)
            ptf3c = nengo.Probe(model.trfm.cconv1.output, synapse=0.005)
            ptf4 = nengo.Probe(model.trfm.output, synapse=0.005)

            probes = gen_probe_list(['trfm io', p0, ptf1, ptf2, ptf4, 0,
                                     'trfm cc', p0, ptf3, ptf3b, ptf3c],
                                    [ptf1, ptf2, ptf3, ptf3b, ptf3c, ptf4])
            probe_list.extend(probes)

            vocab_dict[idstr(ptf1)] = sub_vocab1
            vocab_dict[idstr(ptf2)] = sub_vocab1
            vocab_dict[idstr(ptf3)] = item_vocab
            vocab_dict[idstr(ptf3b)] = pos_vocab
            vocab_dict[idstr(ptf3c)] = sub_vocab2
            vocab_dict[idstr(ptf4)] = sub_vocab1

        if hasattr(model, 'trfm') and \
                not isinstance(model.trfm, TransformationSystemDummy):
            ptf5 = nengo.Probe(model.trfm.am_p1.output, synapse=0.005)
            ptf6 = nengo.Probe(model.trfm.am_n1.output, synapse=0.005)
            ptf7 = nengo.Probe(model.trfm.am_n2.output, synapse=0.005)
            ptf8 = nengo.Probe(model.trfm.am_p2.output, synapse=0.005)

            probes = gen_probe_list(['trfm am1', p0, ptf5, ptf6, 0,
                                     'trfm am2', p0, ptf7, ptf8],
                                    [ptf5, ptf6, ptf7, ptf8])
            probe_list.extend(probes)

            vocab_dict[idstr(ptf5)] = pos_vocab
            vocab_dict[idstr(ptf6)] = item_vocab
            vocab_dict[idstr(ptf7)] = pos1_vocab
            vocab_dict[idstr(ptf8)] = pos1_vocab

        if hasattr(model, 'bg') and True:
            pbg1 = nengo.Probe(model.bg.input, synapse=0.005)
            pbg2 = nengo.Probe(model.bg.output, synapse=0.005)

            probes = gen_probe_list(['bg', p0, pbg1, pbg2])
            probe_list.extend(probes)

        if hasattr(model, 'dec') and True:
            pde1 = nengo.Probe(model.dec.item_dcconv.output, synapse=0.005)
            # pde2 = nengo.Probe(model.dec.select_am, synapse=0.005)
            # pde3 = nengo.Probe(model.dec.select_vis, synapse=0.005)
            pde4 = nengo.Probe(model.dec.am_out, synapse=0.01)
            # pde5 = nengo.Probe(model.dec.vt_out, synapse=0.005)
            pde6 = nengo.Probe(model.dec.pos_mb_gate_sig, synapse=0.005)
            # pde7 = nengo.Probe(model.dec.util_diff_neg, synapse=0.005)
            pde8 = nengo.Probe(model.dec.am_utils, synapse=0.005)
            pde9 = nengo.Probe(model.dec.am2_utils, synapse=0.005)
            # pde10 = nengo.Probe(model.dec.util_diff, synapse=0.005)
            # pde11 = nengo.Probe(model.dec.recall_mb.output, synapse=0.005)
            # pde12 = nengo.Probe(model.dec.dec_am_fr.output, synapse=0.005)
            # pde13 = nengo.Probe(model.dec.dec_am.item_output, synapse=0.005)
            # pde14 = nengo.Probe(model.dec.recall_mb.mem1.output, synapse=0.005)
            pde15 = nengo.Probe(model.dec.output_know, synapse=0.005)
            pde16 = nengo.Probe(model.dec.output_unk, synapse=0.005)
            pde18 = nengo.Probe(model.dec.output_stop, synapse=0.005)
            # pde19 = nengo.Probe(model.dec.am_th_utils, synapse=0.005)
            # pde20 = nengo.Probe(model.dec.fr_th_utils, synapse=0.005)
            pde21 = nengo.Probe(model.dec.dec_output, synapse=0.005)
            # pde22 = nengo.Probe(model.dec.dec_am_fr.input, synapse=0.005)
            # pde23 = nengo.Probe(model.dec.am_def_th_utils, synapse=0.005)
            # pde24 = nengo.Probe(model.dec.fr_def_th_utils, synapse=0.005)
            pde25 = nengo.Probe(model.dec.fr_utils, synapse=0.005)

            probes = gen_probe_list(['dec decconv', pde1, pde4, pde21, 0,
                                     'dec kn unk st', pde15, pde16, pde18, 0,
                                     'dec am utils', pde8, pde9, pde25, 0,
                                     'dec sigs', pde6])
            probe_list.extend(probes)

            vocab_dict[idstr(pde1)] = item_vocab
            vocab_dict[idstr(pde4)] = mtr_vocab
            vocab_dict[idstr(pde21)] = mtr_vocab

        if hasattr(model, 'mtr'):
            pmt1 = nengo.Probe(model.mtr.ramp, synapse=0.005)
            pmt2 = nengo.Probe(model.mtr.ramp_reset_hold, synapse=0.005)
            pmt3 = nengo.Probe(model.mtr.motor_stop_input, synapse=0.005)
            pmt4 = nengo.Probe(model.mtr.motor_init, synapse=0.005)
            pmt5 = nengo.Probe(model.mtr.motor_go, synapse=0.005)

            probes = gen_probe_list(['mtr', p0, pmt1, pmt2, pmt3, pmt4, pmt5])
            probe_list.extend(probes)

    return (probe_list[:-1], vocab_dict)
