import numpy as np

import nengo

from .config import cfg
from .vocabs import vis_vocab, pos_vocab, enum_vocab # noqa
from .vocabs import ps_task_vocab, ps_state_vocab, ps_dec_vocab  # noqa
from .vocabs import mtr_disp_vocab, item_vocab, vocab  # noqa


def idstr(p):
    if not isinstance(p, nengo.Probe):
        return '0'
    else:
        return str(id(p))


def config_and_setup_probes(model):
    config_filename = cfg.probe_data_filename[:-4] + '_cfg.npz'

    version, probe_list, vocab_dict = setup_probes(model)
    config_data = {'sp_dim': cfg.sp_dim, 'probe_list': probe_list,
                   'vocab_dict': vocab_dict, 'prim_vocab': vocab,
                   'version': version}

    np.savez_compressed(config_filename, **config_data)


def setup_probes(model):
    return setup_probes1(model)
    # return setup_probes2(model)
    # return setup_probes3(model)
    # return setup_probes_vis(model)


def setup_probes_vis(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

    version = 1.3

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3])
    probe_list[1] += '*'
    probe_list[3] += '*'

    vocab_dict = {idstr(pvs1): vis_vocab}

    return (version, probe_list, vocab_dict)


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

    version = 1.3
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

    return (version, probe_list, vocab_dict)


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

    version = 1.3
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

    return (version, probe_list, vocab_dict)


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

    version = 1.3
    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3, 0,
                             p0, pps1, pps2, pps3, 0,
                             p0, pps4, pps5, pps6, 0,
                             p0, pps7, pps8, pps9, 0,
                             p0, pmm1, pmm2, pmm3, 0,
                             p0, pmm4, pmm5, pmm6, 0,
                             p0, pmm7, pmm8, pmm9, 0,
                             p0, pbg1, pbg2])
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
                  idstr(pmm7): enum_vocab}

    return (version, probe_list, vocab_dict)
