import os
import numpy as np

import nengo
from nengo.spa import Vocabulary
from nengo.synapses import Lowpass

from .config import cfg
from .vocabs import vis_vocab, pos_vocab, enum_vocab
from .vocabs import ps_task_vocab, ps_state_vocab, ps_dec_vocab, ps_cmp_vocab
from .vocabs import mtr_vocab, mtr_disp_vocab, item_vocab, pos1_vocab, vocab
from .vocabs import mtr_sp_scale_factor
from .modules.working_memory import WorkingMemoryDummy
from .modules.transform_system import TransformationSystemDummy


def idstr(p):
    if not isinstance(p, nengo.Probe):
        return str(p)
    else:
        return str(id(p))


def add_to_graph_list(graph_list, probes, probes_to_legend=[]):
    new_list = map(idstr, probes)

    if not isinstance(probes_to_legend, list):
        probes_to_legend = [probes_to_legend]

    for probe in probes_to_legend:
        new_list[new_list.index(idstr(probe))] += '*'

    if new_list[-1] != '0':
        new_list.append('0')

    graph_list.extend(new_list)

    return new_list


def add_to_vocab_dict(vocab_dict, probe_vocab_map):
    for key in probe_vocab_map:
        vocab_dict[idstr(key)] = probe_vocab_map[key]


def add_to_anim_config(anim_config, key, data_func_name, data_func_params,
                       plot_type_name, plot_type_params):
    data_func_param_dict = {}
    for param_name in data_func_params:
        param = data_func_params[param_name]
        if isinstance(param, nengo.Probe):
            data_func_param_dict[param_name] = idstr(param)
        else:
            data_func_param_dict[param_name] = param

    anim_config.append({'key': key,
                        'data_func': data_func_name,
                        'data_func_params': data_func_param_dict,
                        'plot_type': plot_type_name,
                        'plot_type_params': plot_type_params})


def config_and_setup_probes(model):
    version = 4.0

    config_filename = cfg.probe_data_filename[:-4] + '_cfg.npz'

    graph_list, vocab_dict, anim_config = setup_probes(model)
    config_data = {'sp_dim': cfg.sp_dim, 'graph_list': graph_list,
                   'vocab_dict': vocab_dict, 'prim_vocab': vocab,
                   'anim_config': anim_config,
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

    graph_list = map(idstr, [p0, pvs1, pvs2, pvs3])
    graph_list[1] += '*'
    graph_list[3] += '*'

    vocab_dict = {idstr(pvs1): vis_vocab}

    return (graph_list, vocab_dict)


def setup_probes_generic(model):
    with model:
        model.config[nengo.Probe].synapse = Lowpass(0.005)

        vocab_dict = {}
        graph_list = []
        anim_config = []

        sub_vocab1 = enum_vocab.create_subset(['POS1*ONE', 'POS2*TWO',
                                               'POS3*THR', 'POS4*FOR',
                                               'POS5*FIV'])

        sub_vocab2 = vocab.create_subset(['ADD'])
        sub_vocab2.readonly = False
        sub_vocab2.add('N_ADD', vocab.parse('~ADD'))
        sub_vocab2.add('ADD*ADD', vocab.parse('ADD*ADD'))
        sub_vocab2.add('ADD*ADD*ADD', vocab.parse('ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD', vocab.parse('ADD*ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD*ADD',
        #                vocab.parse('ADD*ADD*ADD*ADD*ADD'))

        sub_vocab3 = vocab.create_subset([])
        sub_vocab3.readonly = False
        sub_vocab3.add('N_POS1*ONE', vocab.parse('~(POS1*ONE)'))
        sub_vocab3.add('N_POS1*TWO', vocab.parse('~(POS1*TWO)'))
        sub_vocab3.add('N_POS1*THR', vocab.parse('~(POS1*THR)'))
        sub_vocab3.add('N_POS1*FOR', vocab.parse('~(POS1*FOR)'))
        sub_vocab3.add('N_POS1*FIV', vocab.parse('~(POS1*FIV)'))
        sub_vocab3.add('ADD', vocab.parse('ADD'))

        if hasattr(model, 'stim'):
            p0 = nengo.Probe(model.stim.output, synapse=None)

            add_to_anim_config(anim_config, key='vis',
                               data_func_name='generic_single',
                               data_func_params={'data': p0},
                               plot_type_name='imshow',
                               plot_type_params={'shape': (28, 28)})
        else:
            p0 = 0

        if hasattr(model, 'vis') and True:
            pvs1 = nengo.Probe(model.vis.output)
            pvs2 = nengo.Probe(model.vis.neg_attention)
            pvs3 = nengo.Probe(model.vis.am_utilities)
            pvs4 = nengo.Probe(model.vis.mb_output)
            pvs5 = nengo.Probe(model.vis.vis_out)

            # probes = gen_graph_list(['vis', p0, pvs1, pvs2, pvs3])
            # vocab_dict[idstr(pvs1)] = vis_vocab

            add_to_graph_list(graph_list, ['vis', p0, pvs1, pvs2, pvs3, 0,
                                           'vis net', pvs4, pvs5])
            add_to_vocab_dict(vocab_dict, {pvs1: vis_vocab})

        # ############ FOR DEBUGGING VIS DETECT SYSTEM ########################
        # if hasattr(model, 'vis') and True:
        #     pvsd1 = nengo.Probe(model.vis.detect_change_net.input_diff)
        #     pvsd2 = nengo.Probe(model.vis.detect_change_net.item_detect)
        #     pvsd3 = nengo.Probe(model.vis.detect_change_net.blank_detect)

        #     probes = gen_graph_list(['vis detect', p0, pvsd1, pvsd2, pvsd3])
        #     graph_list.extend(probes)

        if hasattr(model, 'ps') and True:
            pps1 = nengo.Probe(model.ps.task)
            pps2 = nengo.Probe(model.ps.state)
            pps3 = nengo.Probe(model.ps.dec)

            pps4 = nengo.Probe(model.ps.ps_task_mb.mem1.output)
            pps5 = nengo.Probe(model.ps.ps_task_mb.mem2.output)
            pps6 = nengo.Probe(model.ps.ps_task_mb.mem1.input, synapse=None)
            pps6b = nengo.Probe(model.ps.task_init.output)

            pps7 = nengo.Probe(model.ps.ps_state_mb.mem1.output)
            pps8 = nengo.Probe(model.ps.ps_state_mb.mem2.output)
            pps9 = nengo.Probe(model.ps.ps_state_mb.mem1.input, synapse=None)

            pps10 = nengo.Probe(model.ps.ps_dec_mb.mem1.output)
            pps11 = nengo.Probe(model.ps.ps_dec_mb.mem2.output)
            pps12 = nengo.Probe(model.ps.ps_dec_mb.mem1.input, synapse=None)

            pps13 = nengo.Probe(model.ps.ps_task_mb.gate)
            pps14 = nengo.Probe(model.ps.ps_state_mb.gate)
            pps15 = nengo.Probe(model.ps.ps_dec_mb.gate)

            # probes = gen_graph_list(['ps', p0, pps1, pps2, pps3, 0,
            #                          'ps_task', p0, pps1, pps6, pps4, pps5, pps6b, pps13, 0,  # noqa
            #                          'ps_state', p0, pps2, pps9, pps7, pps8, pps14, 0,  # noqa
            #                          'ps_dec', p0, pps3, pps12, pps10, pps11, pps15],  # noqa
            #                         [pps1, pps2, pps3, pps4, pps5, pps6])
            # graph_list.extend(probes)
            # vocab_dict[idstr(pps1)] = ps_task_vocab
            # vocab_dict[idstr(pps2)] = ps_state_vocab
            # vocab_dict[idstr(pps3)] = ps_dec_vocab
            # vocab_dict[idstr(pps4)] = ps_task_vocab
            # vocab_dict[idstr(pps5)] = ps_task_vocab
            # vocab_dict[idstr(pps6)] = ps_task_vocab
            # vocab_dict[idstr(pps7)] = ps_state_vocab
            # vocab_dict[idstr(pps8)] = ps_state_vocab
            # vocab_dict[idstr(pps9)] = ps_state_vocab
            # vocab_dict[idstr(pps10)] = ps_dec_vocab
            # vocab_dict[idstr(pps11)] = ps_dec_vocab
            # vocab_dict[idstr(pps12)] = ps_dec_vocab

            add_to_graph_list(graph_list,
                              ['ps', p0, pps1, pps2, pps3, 0,
                               'ps_task', p0, pps1, pps6, pps4, pps5, pps6b, pps13, 0,  # noqa
                               'ps_state', p0, pps2, pps9, pps7, pps8, pps14, 0,  # noqa
                               'ps_dec', p0, pps3, pps12, pps10, pps11, pps15],  # noqa
                              [pps1, pps2, pps3, pps4, pps5, pps6])
            add_to_vocab_dict(vocab_dict, {pps1: ps_task_vocab,
                                           pps2: ps_state_vocab,
                                           pps3: ps_dec_vocab,
                                           pps4: ps_task_vocab,
                                           pps5: ps_task_vocab,
                                           pps6: ps_task_vocab,
                                           pps7: ps_state_vocab,
                                           pps8: ps_state_vocab,
                                           pps9: ps_state_vocab,
                                           pps10: ps_dec_vocab,
                                           pps11: ps_dec_vocab,
                                           pps12: ps_dec_vocab})

        if hasattr(model, 'enc') and True:
            pen1 = nengo.Probe(model.enc.pos_inc.gate)
            # pen2 = nengo.Probe(model.enc.pos_mb.gateX)
            # pen3 = nengo.Probe(model.enc.pos_mb.gateN)
            pen4 = nengo.Probe(model.enc.pos_inc.output)
            # pen5 = nengo.Probe(model.enc.pos_mb.mem1.output)
            # pen6 = nengo.Probe(model.enc.pos_mb.am.output)
            pen6 = nengo.Probe(model.enc.pos_inc.reset)
            pen7 = nengo.Probe(model.enc.pos_mb_acc.output)
            pen8 = nengo.Probe(model.enc.pos_output)

            # probes = gen_graph_list(['enc', p0, pen4, pen7, pen6, 0,
            #                          'enc gate', pen1, pen2, pen3],
            #                         [pen4, pen7])
            # graph_list.extend(probes)
            # vocab_dict[idstr(pen4)] = pos_vocab
            # vocab_dict[idstr(pen7)] = pos_vocab

            add_to_graph_list(graph_list,
                              ['enc', p0, pen4, pen7, pen6, pen8, 0,
                               'enc gate', pen1],
                              [pen4, pen7, pen8])
            add_to_vocab_dict(vocab_dict, {pen4: pos_vocab,
                                           pen7: pos_vocab,
                                           pen8: pos_vocab})

        if hasattr(model, 'mem') and True:
            pmm1 = nengo.Probe(model.mem.mb1)
            pmm1a = nengo.Probe(model.mem.mb1)
            pmm2 = nengo.Probe(model.mem.mb1_net.gate)
            pmm3 = nengo.Probe(model.mem.mb1_net.reset)
            pmm4 = nengo.Probe(model.mem.mb2)
            pmm5 = nengo.Probe(model.mem.mb2_net.gate)
            pmm6 = nengo.Probe(model.mem.mb2_net.reset)
            pmm7 = nengo.Probe(model.mem.mb3)
            pmm8 = nengo.Probe(model.mem.mb3_net.gate)
            pmm9 = nengo.Probe(model.mem.mb3_net.reset)

            # probes = gen_graph_list(['mb1', p0, pmm1, pmm2, pmm3, 0,
            #                          'mb2', p0, pmm4, pmm5, pmm6, 0,
            #                          'mb3', p0, pmm7, pmm8, pmm9],
            #                         [pmm1, pmm4, pmm7])
            # graph_list.extend(probes)
            # vocab_dict[idstr(pmm1)] = sub_vocab1
            # vocab_dict[idstr(pmm4)] = sub_vocab1
            # vocab_dict[idstr(pmm7)] = sub_vocab1

            # add_to_graph_list(graph_list,
            #                   ['mb1', p0, pmm1, pmm2, pmm3, 0,
            #                    'mb2', p0, pmm4, pmm5, pmm6, 0,
            #                    'mb3', p0, pmm7, pmm8, pmm9],
            #                   [pmm1, pmm4, pmm7])
            add_to_graph_list(graph_list,
                              ['mb1', p0, pmm1, pmm1a, pmm2, pmm3, 0,
                               'mb2', p0, pmm4, pmm5, pmm6, 0,
                               'mb3', p0, pmm7, pmm8, pmm9])
            add_to_vocab_dict(vocab_dict, {pmm1: sub_vocab1,
                                           pmm4: sub_vocab1,
                                           pmm7: sub_vocab1})

        if hasattr(model, 'mem') and True:
            pmm10 = nengo.Probe(model.mem.mbave_net.input)
            pmm11 = nengo.Probe(model.mem.mbave_net.gate)
            pmm12 = nengo.Probe(model.mem.mbave_net.reset)
            pmm13 = nengo.Probe(model.mem.mbave)
            # pmm14 = nengo.Probe(model.mem.mbave_norm.output)
            # pmm15 = nengo.Probe(model.mem.mbave_in_init.output)

            # probes = gen_graph_list(['mbave', p0, pmm10, pmm11, pmm12, pmm13,
            #                          pmm15, pmm14],
            #                         pmm10)
            # graph_list.extend(probes)
            # vocab_dict[idstr(pmm10)] = sub_vocab2
            # vocab_dict[idstr(pmm13)] = sub_vocab2
            # vocab_dict[idstr(pmm15)] = sub_vocab2

            add_to_graph_list(graph_list,
                              ['mbave', p0, pmm10, pmm11, pmm12, pmm13],
                              [pmm10])
            add_to_vocab_dict(vocab_dict, {pmm10: sub_vocab2,
                                           pmm13: sub_vocab2})

        # if (hasattr(model, 'mem') and not isinstance(model.mem,
        #                                              WorkingMemoryDummy)):
        #     mem = model.mem.mb3a
        #     pmm11 = nengo.Probe(mem.gateX)
        #     pmm12 = nengo.Probe(mem.gateN)
        #     pmm13 = nengo.Probe(mem.gate)
        #     pmm14 = nengo.Probe(mem.reset)
        #     pmm15 = nengo.Probe(model.mem.gate_sig_bias.output)
        #     pmm16 = nengo.Probe(mem.input)

        #     probes = gen_graph_list(['mb3a sigs', p0, pmm11, pmm12, pmm13,
        #                              pmm14, pmm15, pmm16],
        #                             [pmm16])
        #     graph_list.extend(probes)
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

            # probes = gen_graph_list(['trfm io', p0, ptf1, ptf2, ptf4, ptf4b, 0,
            #                          'trfm cc', p0, ptf3, ptf3b, ptf3c, ptf3d, 0, # noqa
            #                          'trfm cmp', ptf5, ptf8, ptf6, ptf9, ptf7,
            #                          ptf10],
            #                         [ptf1, ptf2, ptf3, ptf3b, ptf3c, ptf4,
            #                          ptf4b, ptf6, ptf7])
            # graph_list.extend(probes)

            # vocab_dict[idstr(ptf1)] = sub_vocab1
            # vocab_dict[idstr(ptf2)] = sub_vocab3
            # vocab_dict[idstr(ptf3)] = item_vocab
            # vocab_dict[idstr(ptf3b)] = pos_vocab
            # vocab_dict[idstr(ptf3c)] = sub_vocab2
            # vocab_dict[idstr(ptf3d)] = sub_vocab1
            # vocab_dict[idstr(ptf4)] = sub_vocab2
            # vocab_dict[idstr(ptf4b)] = sub_vocab1
            # vocab_dict[idstr(ptf5)] = ps_cmp_vocab
            # vocab_dict[idstr(ptf6)] = sub_vocab1
            # vocab_dict[idstr(ptf7)] = sub_vocab1
            # vocab_dict[idstr(ptf8)] = sub_vocab1
            # vocab_dict[idstr(ptf9)] = sub_vocab1

            add_to_graph_list(graph_list,
                              ['trfm io', p0, ptf1, ptf2, ptf4, ptf4b, 0,
                               'trfm cc', p0, ptf3, ptf3b, ptf3c, ptf3d, 0, # noqa
                               'trfm cmp', ptf5, ptf8, ptf6, ptf9, ptf7, ptf10], # noqa
                              [ptf1, ptf2, ptf3, ptf3b, ptf3c, ptf4, ptf4b,
                               ptf6, ptf7])
            add_to_vocab_dict(vocab_dict, {ptf1: sub_vocab1,
                                           ptf2: sub_vocab3,
                                           ptf3: item_vocab,
                                           ptf3b: pos_vocab,
                                           ptf3c: sub_vocab2,
                                           ptf3d: sub_vocab1,
                                           ptf4: sub_vocab2,
                                           ptf4b: sub_vocab1,
                                           ptf5: ps_cmp_vocab,
                                           ptf6: sub_vocab1,
                                           ptf7: sub_vocab1,
                                           ptf8: sub_vocab1,
                                           ptf9: sub_vocab1})

        if hasattr(model, 'trfm') and \
                not isinstance(model.trfm, TransformationSystemDummy):
            ptf5 = nengo.Probe(model.trfm.am_trfms.pos1_to_pos)
            ptf6 = nengo.Probe(model.trfm.am_trfms.pos1_to_num)
            ptf7 = nengo.Probe(model.trfm.am_trfms.num_to_pos1)
            ptf8 = nengo.Probe(model.trfm.am_trfms.pos_to_pos1)
            # ptf9 = nengo.Probe(model.trfm.vis_trfm_utils)
            # ptf10 = nengo.Probe(model.trfm.vis_trfm_in)

            # probes = gen_graph_list(['trfm am1', p0, ptf5, ptf6, 0,
            #                          'trfm am2', p0, ptf7, ptf8],
            #                         [ptf5, ptf6, ptf7, ptf8])
            # graph_list.extend(probes)

            # vocab_dict[idstr(ptf5)] = pos_vocab
            # vocab_dict[idstr(ptf6)] = item_vocab
            # vocab_dict[idstr(ptf7)] = pos1_vocab
            # vocab_dict[idstr(ptf8)] = pos1_vocab

            add_to_graph_list(graph_list,
                              ['trfm am1', p0, ptf5, ptf6, 0,
                               'trfm am2', p0, ptf7, ptf8, 0],
                              # 'trfm vis', p0, ptf9, ptf10],
                              [ptf5, ptf6, ptf7, ptf8])
            add_to_vocab_dict(vocab_dict, {ptf5: pos_vocab,
                                           ptf6: item_vocab,
                                           ptf7: pos1_vocab,
                                           ptf8: pos1_vocab})

        if hasattr(model, 'bg') and True:
            pbg1 = nengo.Probe(model.bg.input)
            pbg2 = nengo.Probe(model.bg.output)

            # probes = gen_graph_list(['bg', p0, pbg1, pbg2])
            # graph_list.extend(probes)

            add_to_graph_list(graph_list, ['bg', p0, pbg1, pbg2])

        if hasattr(model, 'dec') and True:
            pde1 = nengo.Probe(model.dec.item_dcconv)
            # pde2 = nengo.Probe(model.dec.select_am)
            # pde3 = nengo.Probe(model.dec.select_vis)
            pde4 = nengo.Probe(model.dec.am_out, synapse=0.01)
            # pde5 = nengo.Probe(model.dec.vt_out)
            pde6 = nengo.Probe(model.dec.pos_mb_gate_sig.output)
            # pde7 = nengo.Probe(model.dec.util_diff_neg)
            pde8 = nengo.Probe(model.dec.am_utils)
            pde9 = nengo.Probe(model.dec.am2_utils)
            pde10 = nengo.Probe(model.dec.util_diff)
            pde11 = nengo.Probe(model.dec.pos_recall_mb)
            # pde12 = nengo.Probe(model.dec.recall_mb.gateX)
            # pde13 = nengo.Probe(model.dec.recall_mb.gateN)
            # pde14a = nengo.Probe(model.dec.recall_mb.mem1.input)
            # pde14b = nengo.Probe(model.dec.recall_mb.mem1.output)
            # pde14c = nengo.Probe(model.dec.recall_mb.mem2.input)
            # pde14d = nengo.Probe(model.dec.recall_mb.mem2.output)
            # pde14e = nengo.Probe(model.dec.recall_mb.mem1.diff.output)
            # pde14f = nengo.Probe(model.dec.recall_mb.reset)
            # pde14g = nengo.Probe(model.dec.recall_mb.mem1.gate)
            # pde12 = nengo.Probe(model.dec.dec_am_fr.output)
            # pde13 = nengo.Probe(model.dec.dec_am.item_output)
            # pde14 = nengo.Probe(model.dec.recall_mb.mem1.output)
            pde15 = nengo.Probe(model.dec.output_know)
            pde16 = nengo.Probe(model.dec.output_unk)
            pde18 = nengo.Probe(model.dec.output_stop)
            # pde19 = nengo.Probe(model.dec.am_th_utils)
            # pde20 = nengo.Probe(model.dec.fr_th_utils)
            pde21 = nengo.Probe(model.dec.output)
            # pde22 = nengo.Probe(model.dec.dec_am_fr.input)
            # pde23 = nengo.Probe(model.dec.am_def_th_utils)
            # pde24 = nengo.Probe(model.dec.fr_def_th_utils)
            pde25 = nengo.Probe(model.dec.fr_utils)
            pde26 = nengo.Probe(model.dec.pos_mb_gate_bias.output)
            pde27 = nengo.Probe(model.dec.pos_acc_input)
            pde28 = nengo.Probe(model.dec.item_dcconv_a)
            pde29 = nengo.Probe(model.dec.item_dcconv_b)

            pde30 = nengo.Probe(model.dec.sel_signals)
            pde31 = nengo.Probe(model.dec.select_out.input0)
            pde32 = nengo.Probe(model.dec.select_out.input1)
            pde33 = nengo.Probe(model.dec.select_out.input3)

            pde34 = nengo.Probe(model.dec.out_class_sr_y)
            pde35 = nengo.Probe(model.dec.out_class_sr_diff)
            pde36 = nengo.Probe(model.dec.out_class_sr_n)

            sel_out_vocab = Vocabulary(5)
            for n in range(5):
                vec = np.zeros(5)
                vec[n] = 1
                sel_out_vocab.add('SEL%d' % n, vec)

            # probes = gen_graph_list(['dec decconv', pde1, pde4, pde21, 0,
            #                          'dec kn unk st', pde15, pde16, pde18, 0,
            #                          'dec am utils', pde8, pde9, pde25, 0,
            #                          'dec sigs', pde6, pde26, pde11, pde27],
            #                         [pde21])
            # # 'dec mb sigs', pde12, pde13, pde14f, pde14g, pde14a, pde14e, pde14b, pde14c, pde14d],  # noqa
            # graph_list.extend(probes)

            # vocab_dict[idstr(pde1)] = item_vocab
            # vocab_dict[idstr(pde4)] = mtr_vocab
            # vocab_dict[idstr(pde21)] = mtr_vocab
            # vocab_dict[idstr(pde11)] = pos_vocab
            # vocab_dict[idstr(pde27)] = pos_vocab
            # # vocab_dict[idstr(pde14a)] = pos_vocab
            # # vocab_dict[idstr(pde14b)] = pos_vocab
            # # vocab_dict[idstr(pde14c)] = pos_vocab
            # # vocab_dict[idstr(pde14d)] = pos_vocab

            add_to_graph_list(graph_list,
                              ['dec decconv', pde28, pde29, pde1, pde4, pde21, 0,
                               'dec kn unk st', pde15, pde16, pde18, 0,
                               'dec am utils', pde8, pde9, pde10, pde25, 0,
                               'dec sigs', pde6, pde26, pde11, pde27, 0,
                               'dec sel', pde30, pde31, pde32, pde33, 0,
                               'dec out class', pde34, pde35, pde36],
                              [pde21, pde30])
            add_to_vocab_dict(vocab_dict, {pde1: item_vocab,
                                           pde4: mtr_vocab,
                                           pde21: mtr_vocab,
                                           pde11: pos_vocab,
                                           pde27: pos_vocab,
                                           pde28: sub_vocab2,
                                           pde29: pos_vocab,
                                           pde30: sel_out_vocab,
                                           pde31: mtr_disp_vocab,
                                           pde32: mtr_disp_vocab,
                                           pde33: mtr_disp_vocab})

        if hasattr(model, 'mtr'):
            pmt1 = nengo.Probe(model.mtr.ramp)
            pmt2 = nengo.Probe(model.mtr.ramp_reset_hold)
            pmt3 = nengo.Probe(model.mtr.motor_stop_input.output)
            pmt4 = nengo.Probe(model.mtr.motor_init.output)
            pmt5 = nengo.Probe(model.mtr.motor_go)
            pmt6 = nengo.Probe(model.mtr.ramp_sig.stop)
            # pmt6 = nengo.Probe(model.mtr.ramp_int_stop)
            pmt7a = nengo.Probe(model.mtr.arm_px_node)
            pmt7b = nengo.Probe(model.mtr.arm_py_node)
            pmt8 = nengo.Probe(model.mtr.pen_down)
            pmt9 = nengo.Probe(model.mtr.zero_centered_arm_ee_loc,
                               synapse=0.01)
            pmt10 = nengo.Probe(model.mtr.zero_centered_tgt_ee_loc,
                                synapse=0.03)

            add_to_graph_list(graph_list,
                              ['mtr', p0, pmt1, pmt2, pmt6, pmt3, pmt4, pmt5, 0,  # noqa
                               'arm', pmt7a, pmt7b, pmt8])

            add_to_anim_config(anim_config, key='mtr',
                               data_func_name='arm_path',
                               data_func_params={'ee_path_data': pmt9,
                                                 'target_path_data': pmt10,
                                                 'pen_status_data': pmt8},
                                                 # 'arm_posx_data': pmt7a,
                                                 # 'arm_posy_data': pmt7b,
                                                 # 'arm_pos_bias': [cfg.mtr_arm_rest_x_bias, cfg.mtr_arm_rest_y_bias]}, # noqa
                               plot_type_name='arm_path_plot',
                               plot_type_params={'show_tick_labels': True,
                                                 'xlim': (-mtr_sp_scale_factor, mtr_sp_scale_factor),  # noqa
                                                 'ylim': (-mtr_sp_scale_factor, mtr_sp_scale_factor)})  # noqa

        # --------------------- ANIMATION CONFIGURATION -----------------------
        anim_config.append({'subplot_width': 5,
                            'subplot_height': 5,
                            'max_subplot_cols': 4,
                            'generator_func_params': {'t_index_step': 10}})

    return (graph_list[:-1], vocab_dict, anim_config)
