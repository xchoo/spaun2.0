import os
import numpy as np

import nengo
from nengo.synapses import Lowpass

from .config import cfg
from .vocabs import vis_vocab, pos_vocab, enum_vocab
from .vocabs import ps_task_vocab, ps_state_vocab, ps_dec_vocab, ps_cmp_vocab
from .vocabs import mtr_vocab, mtr_disp_vocab, item_vocab, pos1_vocab, vocab
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
    return setup_probes_generic(model)


def setup_probes_vis(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output)
        pvs2 = nengo.Probe(model.vis.neg_attention)
        pvs3 = nengo.Probe(model.vis.am_utilities)

    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3])
    probe_list[1] += '*'
    probe_list[3] += '*'

    vocab_dict = {idstr(pvs1): vis_vocab}

    return (probe_list, vocab_dict)


def setup_probes_generic(model):
    with model:
        model.config[nengo.Probe].synapse = Lowpass(0.005)

        vocab_dict = {}
        probe_list = []

        sub_vocab1 = enum_vocab.create_subset(['POS1*ONE', 'POS1*TWO',
                                               'POS1*THR', 'POS1*FOR',
                                               'POS2*TWO', 'POS2*THR',
                                               'POS1*FIV', 'POS1*ZER',
                                               'POS3*ONE', 'POS3*THR'])

        sub_vocab2 = vocab.create_subset(['ADD'])
        sub_vocab2.add('N_ADD', vocab.parse('~ADD'))
        sub_vocab2.add('ADD*ADD', vocab.parse('ADD*ADD'))
        sub_vocab2.add('ADD*ADD*ADD', vocab.parse('ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD', vocab.parse('ADD*ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD*ADD',
        #                vocab.parse('ADD*ADD*ADD*ADD*ADD'))

        sub_vocab3 = vocab.create_subset([])
        sub_vocab3.add('N_POS1*ONE', vocab.parse('~(POS1*ONE)'))
        sub_vocab3.add('N_POS1*TWO', vocab.parse('~(POS1*TWO)'))
        sub_vocab3.add('N_POS1*THR', vocab.parse('~(POS1*THR)'))
        sub_vocab3.add('N_POS1*FOR', vocab.parse('~(POS1*FOR)'))
        sub_vocab3.add('N_POS1*FIV', vocab.parse('~(POS1*FIV)'))
        sub_vocab3.add('ADD', vocab.parse('ADD'))

        if hasattr(model, 'stim'):
            p0 = nengo.Probe(model.stim.output, synapse=None)
        else:
            p0 = 0

        if hasattr(model, 'vis') and True:
            pvs1 = nengo.Probe(model.vis.output)
            pvs2 = nengo.Probe(model.vis.neg_attention)
            pvs3 = nengo.Probe(model.vis.am_utilities)

            # probes = gen_probe_list(['vis', p0, pvs1, pvs2, pvs3], pvs1)
            probes = gen_probe_list(['vis', p0, pvs1, pvs2, pvs3])
            probe_list.extend(probes)
            vocab_dict[idstr(pvs1)] = vis_vocab

        # ############ FOR DEBUGGING VIS DETECT SYSTEM ########################
        # if hasattr(model, 'vis') and True:
        #     pvsd1 = nengo.Probe(model.vis.detect_change_net.input_diff)
        #     pvsd2 = nengo.Probe(model.vis.detect_change_net.item_detect)
        #     pvsd3 = nengo.Probe(model.vis.detect_change_net.blank_detect)

        #     probes = gen_probe_list(['vis detect', p0, pvsd1, pvsd2, pvsd3])
        #     probe_list.extend(probes)

        if hasattr(model, 'ps') and True:
            pps1 = nengo.Probe(model.ps.task)
            pps2 = nengo.Probe(model.ps.state)
            pps3 = nengo.Probe(model.ps.dec)

            pps4 = nengo.Probe(model.ps.ps_task_mb.mem1.output)
            pps5 = nengo.Probe(model.ps.ps_task_mb.mem2.output)
            pps6 = nengo.Probe(model.ps.ps_task_mb.mem1.input)

            pps7 = nengo.Probe(model.ps.ps_state_mb.mem1.output)
            pps8 = nengo.Probe(model.ps.ps_state_mb.mem2.output)
            pps9 = nengo.Probe(model.ps.ps_state_mb.mem1.input)

            pps10 = nengo.Probe(model.ps.ps_dec_mb.mem1.output)
            pps11 = nengo.Probe(model.ps.ps_dec_mb.mem2.output)
            pps12 = nengo.Probe(model.ps.ps_dec_mb.mem1.input)

            pps13 = nengo.Probe(model.ps.ps_task_mb.gate)
            pps14 = nengo.Probe(model.ps.ps_state_mb.gate)
            pps15 = nengo.Probe(model.ps.ps_dec_mb.gate)

            probes = gen_probe_list(['ps', p0, pps1, pps2, pps3, 0,
                                     'ps_task', p0, pps1, pps6, pps4, pps5, pps13, 0,  # noqa
                                     'ps_state', p0, pps2, pps9, pps7, pps8, pps14, 0,  # noqa
                                     'ps_dec', p0, pps3, pps12, pps10, pps11, pps15],  # noqa
                                    [pps1, pps2, pps3, pps4, pps5, pps6])
            probe_list.extend(probes)

            vocab_dict[idstr(pps1)] = ps_task_vocab
            vocab_dict[idstr(pps2)] = ps_state_vocab
            vocab_dict[idstr(pps3)] = ps_dec_vocab
            vocab_dict[idstr(pps4)] = ps_task_vocab
            vocab_dict[idstr(pps5)] = ps_task_vocab
            vocab_dict[idstr(pps6)] = ps_task_vocab
            vocab_dict[idstr(pps7)] = ps_state_vocab
            vocab_dict[idstr(pps8)] = ps_state_vocab
            vocab_dict[idstr(pps9)] = ps_state_vocab
            vocab_dict[idstr(pps10)] = ps_dec_vocab
            vocab_dict[idstr(pps11)] = ps_dec_vocab
            vocab_dict[idstr(pps12)] = ps_dec_vocab

        if hasattr(model, 'enc') and True:
            pen1 = nengo.Probe(model.enc.pos_mb.gate)
            # pen2 = nengo.Probe(model.enc.pos_mb.gateX)
            # pen3 = nengo.Probe(model.enc.pos_mb.gateN)
            pen4 = nengo.Probe(model.enc.pos_mb.mem2.output)
            pen5 = nengo.Probe(model.enc.pos_mb.mem1.output)
            # pen6 = nengo.Probe(model.enc.pos_mb.am.output)
            pen6 = nengo.Probe(model.enc.pos_mb.reset)

            probes = gen_probe_list(['enc', p0, pen1, pen4, pen5, pen6],
                                    [pen4, pen5])
            probe_list.extend(probes)

            vocab_dict[idstr(pen4)] = pos_vocab
            vocab_dict[idstr(pen5)] = pos_vocab

        if hasattr(model, 'mem') and True:
            pmm1 = nengo.Probe(model.mem.mb1)
            pmm2 = nengo.Probe(model.mem.mb1_gate)
            pmm3 = nengo.Probe(model.mem.mb1_reset)
            pmm4 = nengo.Probe(model.mem.mb2)
            pmm5 = nengo.Probe(model.mem.mb2_gate)
            pmm6 = nengo.Probe(model.mem.mb2_reset)
            pmm7 = nengo.Probe(model.mem.mb3)
            pmm8 = nengo.Probe(model.mem.mb3_gate)
            pmm9 = nengo.Probe(model.mem.mb3_reset)

            probes = gen_probe_list(['mb1', p0, pmm1, pmm2, pmm3, 0,
                                     'mb2', p0, pmm4, pmm5, pmm6, 0,
                                     'mb3', p0, pmm7, pmm8, pmm9],
                                    [pmm1, pmm4, pmm7])
            probe_list.extend(probes)
            vocab_dict[idstr(pmm1)] = sub_vocab1
            vocab_dict[idstr(pmm4)] = sub_vocab1
            vocab_dict[idstr(pmm7)] = sub_vocab1

        if hasattr(model, 'mem') and True:
            pmm10 = nengo.Probe(model.mem.mbave_in)
            pmm11 = nengo.Probe(model.mem.mbave_gate)
            pmm12 = nengo.Probe(model.mem.mbave_reset)
            pmm13 = nengo.Probe(model.mem.mbave)
            pmm14 = nengo.Probe(model.mem.mbave_norm.output)
            pmm15 = nengo.Probe(model.mem.mbave_in_init.output)

            probes = gen_probe_list(['mbave', p0, pmm10, pmm11, pmm12, pmm13,
                                     pmm15, pmm14],
                                    pmm10)
            probe_list.extend(probes)
            vocab_dict[idstr(pmm10)] = sub_vocab2
            vocab_dict[idstr(pmm13)] = sub_vocab2
            vocab_dict[idstr(pmm15)] = sub_vocab2

        # if (hasattr(model, 'mem') and not isinstance(model.mem,
        #                                              WorkingMemoryDummy)):
        #     mem = model.mem.mb3a
        #     pmm11 = nengo.Probe(mem.gateX)
        #     pmm12 = nengo.Probe(mem.gateN)
        #     pmm13 = nengo.Probe(mem.gate)
        #     pmm14 = nengo.Probe(mem.reset)
        #     pmm15 = nengo.Probe(model.mem.gate_sig_bias.output)
        #     pmm16 = nengo.Probe(mem.input)

        #     probes = gen_probe_list(['mb3a sigs', p0, pmm11, pmm12, pmm13,
        #                              pmm14, pmm15, pmm16],
        #                             [pmm16])
        #     probe_list.extend(probes)
        #     vocab_dict[idstr(pmm16)] = sub_vocab1

        if hasattr(model, 'trfm') and \
                not isinstance(model.trfm, TransformationSystemDummy):
            ptf1 = nengo.Probe(model.trfm.select_in_a.output)
            ptf2 = nengo.Probe(model.trfm.select_in_b.output)
            ptf3 = nengo.Probe(model.trfm.cconv1.output)
            ptf3b = nengo.Probe(model.trfm.cconv1.output)
            ptf3c = nengo.Probe(model.trfm.cconv1.output)
            ptf3d = nengo.Probe(model.trfm.cconv1.output)
            ptf4 = nengo.Probe(model.trfm.output)
            ptf4b = nengo.Probe(model.trfm.output)
            ptf5 = nengo.Probe(model.trfm.compare.output)
            ptf6 = nengo.Probe(model.trfm.norm_a.output)
            ptf7 = nengo.Probe(model.trfm.norm_b.output)
            ptf8 = nengo.Probe(model.trfm.norm_a.input)
            ptf9 = nengo.Probe(model.trfm.norm_b.input)
            ptf10 = nengo.Probe(model.trfm.compare.dot_prod)

            probes = gen_probe_list(['trfm io', p0, ptf1, ptf2, ptf4, ptf4b, 0,
                                     'trfm cc', p0, ptf3, ptf3b, ptf3c, ptf3d, 0, # noqa
                                     'trfm cmp', ptf5, ptf8, ptf6, ptf9, ptf7,
                                     ptf10],
                                    [ptf1, ptf2, ptf3, ptf3b, ptf3c, ptf4,
                                     ptf4b, ptf6, ptf7])
            probe_list.extend(probes)

            vocab_dict[idstr(ptf1)] = sub_vocab1
            vocab_dict[idstr(ptf2)] = sub_vocab3
            vocab_dict[idstr(ptf3)] = item_vocab
            vocab_dict[idstr(ptf3b)] = pos_vocab
            vocab_dict[idstr(ptf3c)] = sub_vocab2
            vocab_dict[idstr(ptf3d)] = sub_vocab1
            vocab_dict[idstr(ptf4)] = sub_vocab2
            vocab_dict[idstr(ptf4b)] = sub_vocab1
            vocab_dict[idstr(ptf5)] = ps_cmp_vocab
            vocab_dict[idstr(ptf6)] = sub_vocab1
            vocab_dict[idstr(ptf7)] = sub_vocab1
            vocab_dict[idstr(ptf8)] = sub_vocab1
            vocab_dict[idstr(ptf9)] = sub_vocab1

        if hasattr(model, 'trfm') and \
                not isinstance(model.trfm, TransformationSystemDummy):
            ptf5 = nengo.Probe(model.trfm.am_p1.output)
            ptf6 = nengo.Probe(model.trfm.am_n1.output)
            ptf7 = nengo.Probe(model.trfm.am_n2.output)
            ptf8 = nengo.Probe(model.trfm.am_p2.output)

            probes = gen_probe_list(['trfm am1', p0, ptf5, ptf6, 0,
                                     'trfm am2', p0, ptf7, ptf8],
                                    [ptf5, ptf6, ptf7, ptf8])
            probe_list.extend(probes)

            vocab_dict[idstr(ptf5)] = pos_vocab
            vocab_dict[idstr(ptf6)] = item_vocab
            vocab_dict[idstr(ptf7)] = pos1_vocab
            vocab_dict[idstr(ptf8)] = pos1_vocab

        if hasattr(model, 'bg') and True:
            pbg1 = nengo.Probe(model.bg.input)
            pbg2 = nengo.Probe(model.bg.output)

            probes = gen_probe_list(['bg', p0, pbg1, pbg2])
            probe_list.extend(probes)

        if hasattr(model, 'dec') and True:
            pde1 = nengo.Probe(model.dec.item_dcconv.output)
            # pde2 = nengo.Probe(model.dec.select_am)
            # pde3 = nengo.Probe(model.dec.select_vis)
            pde4 = nengo.Probe(model.dec.am_out, synapse=0.01)
            # pde5 = nengo.Probe(model.dec.vt_out)
            pde6 = nengo.Probe(model.dec.pos_mb_gate_sig.output)
            # pde7 = nengo.Probe(model.dec.util_diff_neg)
            pde8 = nengo.Probe(model.dec.am_utils)
            pde9 = nengo.Probe(model.dec.am2_utils)
            # pde10 = nengo.Probe(model.dec.util_diff)
            # pde11 = nengo.Probe(model.dec.recall_mb.output)
            # pde12 = nengo.Probe(model.dec.dec_am_fr.output)
            # pde13 = nengo.Probe(model.dec.dec_am.item_output)
            # pde14 = nengo.Probe(model.dec.recall_mb.mem1.output)
            pde15 = nengo.Probe(model.dec.output_know)
            pde16 = nengo.Probe(model.dec.output_unk)
            pde18 = nengo.Probe(model.dec.output_stop.output)
            # pde19 = nengo.Probe(model.dec.am_th_utils)
            # pde20 = nengo.Probe(model.dec.fr_th_utils)
            pde21 = nengo.Probe(model.dec.dec_output)
            # pde22 = nengo.Probe(model.dec.dec_am_fr.input)
            # pde23 = nengo.Probe(model.dec.am_def_th_utils)
            # pde24 = nengo.Probe(model.dec.fr_def_th_utils)
            pde25 = nengo.Probe(model.dec.fr_utils)
            pde26 = nengo.Probe(model.dec.pos_mb_gate_bias.output)

            probes = gen_probe_list(['dec decconv', pde1, pde4, pde21, 0,
                                     'dec kn unk st', pde15, pde16, pde18, 0,
                                     'dec am utils', pde8, pde9, pde25, 0,
                                     'dec sigs', pde6, pde26],
                                     [pde21])
            probe_list.extend(probes)

            vocab_dict[idstr(pde1)] = item_vocab
            vocab_dict[idstr(pde4)] = mtr_vocab
            vocab_dict[idstr(pde21)] = mtr_vocab

        if hasattr(model, 'mtr'):
            pmt1 = nengo.Probe(model.mtr.ramp)
            pmt2 = nengo.Probe(model.mtr.ramp_reset_hold.output)
            pmt3 = nengo.Probe(model.mtr.motor_stop_input.output)
            pmt4 = nengo.Probe(model.mtr.motor_init.output)
            pmt5 = nengo.Probe(model.mtr.motor_go)

            probes = gen_probe_list(['mtr', p0, pmt1, pmt2, pmt3, pmt4, pmt5])
            probe_list.extend(probes)

    return (probe_list[:-1], vocab_dict)
