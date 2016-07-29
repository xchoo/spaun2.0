import os
import numpy as np

import nengo
from nengo.spa import Vocabulary
from nengo.synapses import Lowpass

from .configurator import cfg
from .vocabulator import vocab
from .experimenter import experiment
from .modules.working_memory import WorkingMemoryDummy
from .modules.transform_system import TransformationSystemDummy
from .modules.motor.data import mtr_data


def idstr(p):
    if not isinstance(p, nengo.Probe):
        return str(p)
    else:
        return str(id(p))


def add_to_graph_list(graph_list, probes, probes_to_legend=[]):
    new_list = list(map(idstr, probes))

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


def config_and_setup_probes(model, probe_data_dir, probe_data_filename,
                            setup_probes_func=None):
    version = 4.0

    config_filename = probe_data_filename[:-4] + '_cfg.npz'

    if setup_probes_func is None:
        setup_probes_func = setup_probes

    graph_list, vocab_dict, anim_config = setup_probes_func(model)
    config_data = {'sp_dim': vocab.sp_dim, 'graph_list': graph_list,
                   'vocab_dict': vocab_dict, 'prim_vocab': vocab,
                   'anim_config': anim_config,
                   'dt': cfg.sim_dt, 'version': version}

    np.savez_compressed(os.path.join(probe_data_dir, config_filename),
                        **config_data)


def write_probe_data(sim, probe_data_dir, probe_data_filename):
    # Generic probe data (time and stimulus sequence)
    probe_data = {'trange': sim.trange(),
                  'stim_seq': experiment.stim_seq_list}

    # Sort out the actual probes from sim
    for probe in sim.data.keys():
        if isinstance(probe, nengo.Probe):
            probe_data[idstr(probe)] = sim.data[probe]
    np.savez_compressed(os.path.join(probe_data_dir,
                                     probe_data_filename),
                        **probe_data)


def setup_probes(model):
    return setup_probes_generic(model)


def setup_probes_vis(model):
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output)
        pvs2 = nengo.Probe(model.vis.neg_attention)
        pvs3 = nengo.Probe(model.vis.am_utilities)

    graph_list = list(map(idstr, [p0, pvs1, pvs2, pvs3]))
    graph_list[1] += '*'
    graph_list[3] += '*'

    vocab_dict = {idstr(pvs1): vocab.vis_main}

    return (graph_list, vocab_dict, [])


def setup_probes_animation(model):
    with model:
        model.config[nengo.Probe].synapse = Lowpass(0.005)

        anim_config = []

        if hasattr(model, 'stim') and hasattr(model, 'vis'):
            # -------------------- VISION STIMULI PROBES ----------------------
            p0 = nengo.Probe(model.stim.output, synapse=None)

            add_to_anim_config(anim_config, key='vis',
                               data_func_name='generic_single',
                               data_func_params={'data': p0},
                               plot_type_name='imshow',
                               plot_type_params={'shape': (28, 28)})

            # --------------------- MOTOR OUTPUT PROBES -----------------------
            pmt1 = nengo.Probe(model.mtr.pen_down,
                               synapse=0.05)
            pmt2 = nengo.Probe(model.mtr.zero_centered_arm_ee_loc,
                               synapse=0.05)
            pmt3 = nengo.Probe(model.mtr.zero_centered_tgt_ee_loc,
                               synapse=0.05)
            # pmt4a = nengo.Probe(model.mtr.arm_px_node)
            # pmt4b = nengo.Probe(model.mtr.arm_py_node)

            add_to_anim_config(anim_config, key='mtr',
               data_func_name='arm_path',
               data_func_params={'ee_path_data': pmt2,
                                 'target_path_data': pmt3,
                                 'pen_status_data': pmt1},
                                 # 'arm_posx_data': pmt4a,
                                 # 'arm_posy_data': pmt4b,
                                 # 'arm_pos_bias': [cfg.mtr_arm_rest_x_bias,
                                 #                  cfg.mtr_arm_rest_y_bias]},
               plot_type_name='arm_path_plot',
               plot_type_params={'show_tick_labels': True,
                                 'xlim': (-mtr_data.sp_scaling_factor,
                                          mtr_data.sp_scaling_factor),
                                 'ylim': (-mtr_data.sp_scaling_factor,
                                          mtr_data.sp_scaling_factor)})  # noqa

            # --------------------- ANIMATION CONFIGURATION -------------------
            anim_config.append({'subplot_width': 5,
                                'subplot_height': 5,
                                'max_subplot_cols': 4,
                                'generator_func_params': {'t_index_step': 10}})
        else:
            raise RuntimeError('Unable to setup animation probes. Spaun ' +
                               '`vis` and `mtr` modules are required.')

        return ([], {}, anim_config)


def setup_probes_generic(model):
    with model:
        model.config[nengo.Probe].synapse = Lowpass(0.005)

        vocab_dict = {}
        graph_list = []

        sub_vocab1 = vocab.enum.create_subset(['POS1*ONE', 'POS2*TWO',
                                               'POS3*THR', 'POS4*FOR',
                                               'POS5*FIV'])

        sub_vocab2 = vocab.main.create_subset(['ADD'])
        sub_vocab2.readonly = False
        sub_vocab2.add('N_ADD', vocab.main.parse('~ADD'))
        sub_vocab2.add('ADD*ADD', vocab.main.parse('ADD*ADD'))
        sub_vocab2.add('ADD*ADD*ADD', vocab.main.parse('ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD', vocab.main.parse('ADD*ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD*ADD',
        #                vocab.main.parse('ADD*ADD*ADD*ADD*ADD'))

        sub_vocab3 = vocab.main.create_subset([])
        sub_vocab3.readonly = False
        # sub_vocab3.add('N_POS1*ONE', vocab.main.parse('~(POS1*ONE)'))
        # sub_vocab3.add('N_POS1*TWO', vocab.main.parse('~(POS1*TWO)'))
        # sub_vocab3.add('N_POS1*THR', vocab.main.parse('~(POS1*THR)'))
        # sub_vocab3.add('N_POS1*FOR', vocab.main.parse('~(POS1*FOR)'))
        # sub_vocab3.add('N_POS1*FIV', vocab.main.parse('~(POS1*FIV)'))
        sub_vocab3.add('ADD', vocab.main.parse('ADD'))
        sub_vocab3.add('INC', vocab.main.parse('INC'))

        vocab_seq_list = vocab.main.create_subset([])
        vocab_seq_list.readonly = False
        for sp_str in ['POS1*ONE', 'POS2*TWO', 'POS3*THR', 'POS4*FOR',
                       'POS5*FIV', 'POS6*SIX', 'POS7*SEV', 'POS8*EIG']:
            vocab_seq_list.add(sp_str, vocab.main.parse(sp_str))

        vocab_rpm = vocab.main.create_subset([])
        vocab_rpm.readonly = False
        for i in [1, 3, 8]:
            sp_str = vocab.num_sp_strs[i]
            vocab_rpm.add('A_(P1+P2+P3)*%s' % sp_str,
                          vocab.main.parse('POS1*%s+POS2*%s+POS3*%s' %
                                           (sp_str, sp_str, sp_str)))
            vocab_rpm.add('N_(P1+P2+P3)*%s' % sp_str,
                          vocab.main.parse('~(POS1*%s+POS2*%s+POS3*%s)' %
                                           (sp_str, sp_str, sp_str)))

        vocab_pos1 = vocab.main.create_subset([])
        vocab_pos1.readonly = False
        for sp_str in vocab.num_sp_strs:
            p1_str = 'POS1*%s' % sp_str
            vocab_pos1.add(p1_str, vocab.main.parse(p1_str))

        mem_vocab = vocab_seq_list
        # mem_vocab = vocab_pos1

        ####
        vocab_seq_list = vocab_rpm

        if hasattr(model, 'stim'):
            p0 = nengo.Probe(model.stim.output, synapse=None)
        else:
            p0 = 0

        if hasattr(model, 'vis') and True:
            pvs1 = nengo.Probe(model.vis.output)
            pvs2 = nengo.Probe(model.vis.neg_attention)
            pvs3 = nengo.Probe(model.vis.am_utilities)
            pvs4 = nengo.Probe(model.vis.mb_output)
            pvs5 = nengo.Probe(model.vis.vis_out)

            # probes = gen_graph_list(['vis', p0, pvs1, pvs2, pvs3])
            # vocab_dict[idstr(pvs1)] = vocab.vis_main

            add_to_graph_list(graph_list, ['vis', p0, pvs1, pvs2, pvs3, 0,
                                           'vis net', pvs4, pvs5])
            add_to_vocab_dict(vocab_dict, {pvs1: vocab.vis_main})

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

            pps4 = nengo.Probe(model.ps.task_mb.mem1.output)
            pps5 = nengo.Probe(model.ps.task_mb.mem2.output)
            pps6 = nengo.Probe(model.ps.task_mb.mem1.input, synapse=None)
            pps6b = nengo.Probe(model.ps.task_init.output)

            pps7 = nengo.Probe(model.ps.state_mb.mem1.output)
            pps8 = nengo.Probe(model.ps.state_mb.mem2.output)
            pps9 = nengo.Probe(model.ps.state_mb.mem1.input, synapse=None)

            pps10 = nengo.Probe(model.ps.dec_mb.mem1.output)
            pps11 = nengo.Probe(model.ps.dec_mb.mem2.output)
            pps12 = nengo.Probe(model.ps.dec_mb.mem1.input, synapse=None)

            pps13 = nengo.Probe(model.ps.task_mb.gate)
            pps14 = nengo.Probe(model.ps.state_mb.gate)
            pps15 = nengo.Probe(model.ps.dec_mb.gate)
            pps13b = nengo.Probe(model.ps.task_mb.mem1.gate)
            pps14b = nengo.Probe(model.ps.state_mb.mem1.gate)
            pps15b = nengo.Probe(model.ps.dec_mb.mem1.gate)

            # pps13g = nengo.Probe(model.ps.task_mb.mem2.gate)
            # pps14g = nengo.Probe(model.ps.state_mb.mem2.gate)
            # pps15g = nengo.Probe(model.ps.dec_mb.mem2.gate)

            # pps13d = nengo.Probe(model.ps.task_mb.mem1.diff.output)
            # pps14d = nengo.Probe(model.ps.state_mb.mem1.diff.output)
            # pps15d = nengo.Probe(model.ps.dec_mb.mem1.diff.output)

            # pps13s = nengo.Probe(model.ps.task_mb.mem1.reset_gate.output)
            # pps14s = nengo.Probe(model.ps.state_mb.mem1.reset_gate.output)
            # pps15s = nengo.Probe(model.ps.dec_mb.mem1.reset_gate.output)

            pps13r = nengo.Probe(model.ps.task_mb.reset)
            pps14r = nengo.Probe(model.ps.state_mb.reset)
            pps15r = nengo.Probe(model.ps.dec_mb.reset)

            pps16 = nengo.Probe(model.ps.action)
            pps17 = nengo.Probe(model.ps.action_in)

            # probes = gen_graph_list(['ps', p0, pps1, pps2, pps3, 0,
            #                          'ps_task', p0, pps1, pps6, pps4, pps5, pps6b, pps13, 0,  # noqa
            #                          'ps_state', p0, pps2, pps9, pps7, pps8, pps14, 0,  # noqa
            #                          'ps_dec', p0, pps3, pps12, pps10, pps11, pps15],  # noqa
            #                         [pps1, pps2, pps3, pps4, pps5, pps6])
            # graph_list.extend(probes)
            # vocab_dict[idstr(pps1)] = vocab.ps_task
            # vocab_dict[idstr(pps2)] = vocab.ps_state
            # vocab_dict[idstr(pps3)] = vocab.ps_dec
            # vocab_dict[idstr(pps4)] = vocab.ps_task
            # vocab_dict[idstr(pps5)] = vocab.ps_task
            # vocab_dict[idstr(pps6)] = vocab.ps_task
            # vocab_dict[idstr(pps7)] = vocab.ps_state
            # vocab_dict[idstr(pps8)] = vocab.ps_state
            # vocab_dict[idstr(pps9)] = vocab.ps_state
            # vocab_dict[idstr(pps10)] = vocab.ps_dec
            # vocab_dict[idstr(pps11)] = vocab.ps_dec
            # vocab_dict[idstr(pps12)] = vocab.ps_dec

            add_to_graph_list(graph_list,
                              ['ps', p0, pps1, pps2, pps3, 0,
                               # 'ps_task', p0, pps1, pps6, pps4, pps5, pps6b, pps13d, pps13s, pps13, pps13g, pps13r, 0,  # noqa
                               # 'ps_state', p0, pps2, pps9, pps7, pps8, pps14, pps14d, pps14s, pps14g, pps14r, 0,  # noqa
                               # 'ps_dec', p0, pps3, pps12, pps10, pps11, pps15, pps15d, pps15s, pps15g, pps15r, 0,  # noqa
                               'ps_task', p0, pps6, pps4, pps5, pps6b, pps13, pps13b, pps13r, 0,  # noqa
                               'ps_state', p0, pps9, pps7, pps8, pps14, pps14b, pps14r, 0,  # noqa
                               'ps_dec', p0, pps12, pps10, pps11, pps15, pps15b, pps15r, 0,  # noqa
                               'ps_action', p0, pps17, pps16],
                              [pps1, pps2, pps3, pps4, pps5, pps6, pps16])
            add_to_vocab_dict(vocab_dict, {pps1: vocab.ps_task,
                                           pps2: vocab.ps_state,
                                           pps3: vocab.ps_dec,
                                           pps4: vocab.ps_task,
                                           pps5: vocab.ps_task,
                                           pps6: vocab.ps_task,
                                           # pps13d: vocab.ps_task,
                                           # pps13s: vocab.ps_task,
                                           pps7: vocab.ps_state,
                                           pps8: vocab.ps_state,
                                           pps9: vocab.ps_state,
                                           # pps14d: vocab.ps_state,
                                           # pps14s: vocab.ps_state,
                                           pps10: vocab.ps_dec,
                                           pps11: vocab.ps_dec,
                                           pps12: vocab.ps_dec,
                                           # pps15d: vocab.ps_dec,
                                           # pps15s: vocab.ps_dec,
                                           pps16: vocab.ps_action_learn,
                                           pps17: vocab.ps_action_learn})

        if hasattr(model, 'enc') and True:
            pen1 = nengo.Probe(model.enc.pos_inc.gate)
            pen2 = nengo.Probe(model.enc.pos_inc.pos_mb.gateX)
            # pen2 = nengo.Probe(model.enc.pos_mb.gateX)
            # pen3 = nengo.Probe(model.enc.pos_mb.gateN)
            pen4 = nengo.Probe(model.enc.pos_inc.output)
            pen5 = nengo.Probe(model.enc.pos_inc.pos_mb.mem1.output)
            pen5b = nengo.Probe(model.enc.pos_inc.pos_mb.mem2.output)
            # pen5 = nengo.Probe(model.enc.pos_mb.mem1.output)
            # pen6 = nengo.Probe(model.enc.pos_mb.am.output)
            pen6 = nengo.Probe(model.enc.pos_inc.reset)
            pen7 = nengo.Probe(model.enc.pos_mb_acc.output)
            pen7a = nengo.Probe(model.enc.pos_mb_acc.input)
            pen8 = nengo.Probe(model.enc.pos_output)

            # probes = gen_graph_list(['enc', p0, pen4, pen7, pen6, 0,
            #                          'enc gate', pen1, pen2, pen3],
            #                         [pen4, pen7])
            # graph_list.extend(probes)
            # vocab_dict[idstr(pen4)] = vocab.pos
            # vocab_dict[idstr(pen7)] = vocab.pos

            add_to_graph_list(graph_list,
                              # ['enc', p0, pen1, pen4, pen7, pen7a, pen6],
                              ['enc', p0, pen1, pen2, pen4, pen5, pen5b, pen6],
                              [pen4])
            add_to_vocab_dict(vocab_dict, {pen4: vocab.pos,
                                           pen5: vocab.pos,
                                           pen5b: vocab.pos,
                                           pen7: vocab.pos,
                                           pen7a: vocab.pos,
                                           pen8: vocab.pos})

        if hasattr(model, 'mem') and True:
            pmm1 = nengo.Probe(model.mem.mb1)
            pmm1a = nengo.Probe(model.mem.mb1_net.mb_reh)
            pmm1b = nengo.Probe(model.mem.mb1_net.mb_dcy)
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
                              ['mb1', p0, pmm1, pmm1a, pmm1b, pmm2, pmm3, 0,
                               'mb2', p0, pmm4, pmm5, pmm6, 0,
                               'mb3', p0, pmm7, pmm8, pmm9])
            add_to_vocab_dict(vocab_dict, {pmm1: mem_vocab,
                                           pmm1a: mem_vocab,
                                           pmm1b: mem_vocab,
                                           pmm4: vocab_seq_list,
                                           pmm7: vocab_seq_list})

        if hasattr(model, 'mem') and True:
            pmm1i = nengo.Probe(model.mem.input)
            pmm1ai = nengo.Probe(model.mem.mb1_net.mba.mem1.input)
            pmm1bi = nengo.Probe(model.mem.mb1_net.mba.mem2.input)
            pmm1g = nengo.Probe(model.mem.mb1_net.gate)
            pmm1gx = nengo.Probe(model.mem.mb1_net.mba.gateX)
            pmm1gn = nengo.Probe(model.mem.mb1_net.mba.gateN)
            pmm1ag = nengo.Probe(model.mem.mb1_net.mba.mem1.gate)
            pmm1bg = nengo.Probe(model.mem.mb1_net.mba.mem2.gate)
            # pmm1a1 = nengo.Probe(model.mem.mb1_net.mba.mem1.output)
            # pmm1a2 = nengo.Probe(model.mem.mb1_net.mba.mem2.output)
            # pmm1b1 = nengo.Probe(model.mem.mb1_net.mbb.mem1.output)
            # pmm1b2 = nengo.Probe(model.mem.mb1_net.mbb.mem2.output)
            # pmm1br = nengo.Probe(model.mem.mb1_net.mbb.mem1.reset)
            # pmm1bg = nengo.Probe(model.mem.mb1_net.mbb.mem1.gate)

            add_to_graph_list(graph_list,
                              ['mb1 details', pmm1i, pmm1ai, pmm1bi, pmm1a, pmm1g, pmm1gx, pmm1gn, pmm1ag, pmm1bg, 0,  # noqa
                               ])

            add_to_vocab_dict(vocab_dict, {pmm1i: mem_vocab,
                                           pmm1ai: mem_vocab,
                                           pmm1bi: mem_vocab})

        if hasattr(model, 'mem') and True:
            pmm10 = nengo.Probe(model.mem.mbave_net.input)
            pmm11 = nengo.Probe(model.mem.mbave_net.gate)
            pmm12 = nengo.Probe(model.mem.mbave_net.reset)
            pmm13 = nengo.Probe(model.mem.mbave)
            pmm13a = nengo.Probe(model.mem.mbave)
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
                              ['mbave', p0, pmm10, pmm11, pmm12, pmm13, pmm13a],  # noqa
                              [pmm10])
            add_to_vocab_dict(vocab_dict, {pmm10: sub_vocab2,
                                           pmm13: sub_vocab3})

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
            ptf11a = nengo.Probe(model.trfm.cconv1.A)
            ptf11b = nengo.Probe(model.trfm.cconv1.B)

            # probes = gen_graph_list(['trfm io', p0, ptf1, ptf2, ptf4, ptf4b, 0,
            #                          'trfm cc', p0, ptf3, ptf3b, ptf3c, ptf3d, 0, # noqa
            #                          'trfm cmp', ptf5, ptf8, ptf6, ptf9, ptf7,
            #                          ptf10],
            #                         [ptf1, ptf2, ptf3, ptf3b, ptf3c, ptf4,
            #                          ptf4b, ptf6, ptf7])
            # graph_list.extend(probes)

            # vocab_dict[idstr(ptf1)] = sub_vocab1
            # vocab_dict[idstr(ptf2)] = sub_vocab3
            # vocab_dict[idstr(ptf3)] = vocab.item
            # vocab_dict[idstr(ptf3b)] = vocab.pos
            # vocab_dict[idstr(ptf3c)] = sub_vocab2
            # vocab_dict[idstr(ptf3d)] = sub_vocab1
            # vocab_dict[idstr(ptf4)] = sub_vocab2
            # vocab_dict[idstr(ptf4b)] = sub_vocab1
            # vocab_dict[idstr(ptf5)] = vocab.ps_cmp
            # vocab_dict[idstr(ptf6)] = sub_vocab1
            # vocab_dict[idstr(ptf7)] = sub_vocab1
            # vocab_dict[idstr(ptf8)] = sub_vocab1
            # vocab_dict[idstr(ptf9)] = sub_vocab1

            add_to_graph_list(graph_list,
                              ['trfm io', p0, ptf1, ptf2, ptf4, 0,
                               'trfm cc', p0, pmm11, ptf3, ptf3b, ptf11a, ptf11b, 0, # noqa
                               'trfm cmp', ptf5, ptf8, ptf6, ptf9, ptf7, ptf10], # noqa
                              [ptf1, ptf4, ptf6, ptf7])
            # add_to_vocab_dict(vocab_dict, {ptf1: sub_vocab1,
            #                                ptf2: sub_vocab3,
            #                                ptf3: vocab.item,
            #                                ptf3b: vocab.pos,
            #                                ptf3c: sub_vocab2,
            #                                ptf3d: sub_vocab1,
            #                                ptf4: sub_vocab2,
            #                                ptf4b: sub_vocab1,
            #                                ptf5: vocab.ps_cmp,
            #                                ptf6: sub_vocab1,
            #                                ptf7: sub_vocab1,
            #                                ptf8: sub_vocab1,
            #                                ptf9: sub_vocab1})
            add_to_vocab_dict(vocab_dict, {ptf1: mem_vocab,
                                           ptf2: mem_vocab,
                                           ptf3: vocab_rpm,
                                           ptf4: mem_vocab,
                                           ptf5: vocab.ps_cmp,
                                           ptf6: sub_vocab1,
                                           ptf7: sub_vocab1,
                                           ptf8: sub_vocab1,
                                           ptf9: sub_vocab1,
                                           ptf11a: vocab_rpm,
                                           ptf11b: vocab_rpm})

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

            # vocab_dict[idstr(ptf5)] = vocab.pos
            # vocab_dict[idstr(ptf6)] = vocab.item
            # vocab_dict[idstr(ptf7)] = vocab.pos1
            # vocab_dict[idstr(ptf8)] = vocab.pos1

            add_to_graph_list(graph_list,
                              ['trfm am1', p0, ptf5, ptf6, 0,
                               'trfm am2', p0, ptf7, ptf8, 0],
                              # 'trfm vis', p0, ptf9, ptf10],
                              [ptf5, ptf6, ptf7, ptf8])
            add_to_vocab_dict(vocab_dict, {ptf5: vocab.pos,
                                           ptf6: vocab.item,
                                           ptf7: vocab.pos1,
                                           ptf8: vocab.pos1})

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
            pde11a = nengo.Probe(model.dec.pos_recall_mb_in)
            pde11b = nengo.Probe(model.dec.fr_recall_mb.gate)
            pde11c = nengo.Probe(model.dec.fr_recall_mb.reset)
            pde11d = nengo.Probe(model.dec.fr_recall_mb.mem1.input)
            pde11e = nengo.Probe(model.dec.fr_recall_mb.mem2.input)
            pde12 = nengo.Probe(model.dec.fr_dcconv.A)
            pde13 = nengo.Probe(model.dec.fr_dcconv.B)
            pde14 = nengo.Probe(model.dec.fr_dcconv.output)

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
            pde39 = nengo.Probe(model.dec.output_classify.fr_utils_n)

            pde37 = nengo.Probe(model.dec.serial_decode.inhibit)
            pde38 = nengo.Probe(model.dec.free_recall_decode.inhibit)

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

            # vocab_dict[idstr(pde1)] = vocab.item
            # vocab_dict[idstr(pde4)] = vocab.mtr
            # vocab_dict[idstr(pde21)] = vocab.mtr
            # vocab_dict[idstr(pde11)] = vocab.pos
            # vocab_dict[idstr(pde27)] = vocab.pos
            # # vocab_dict[idstr(pde14a)] = vocab.pos
            # # vocab_dict[idstr(pde14b)] = vocab.pos
            # # vocab_dict[idstr(pde14c)] = vocab.pos
            # # vocab_dict[idstr(pde14d)] = vocab.pos

            add_to_graph_list(graph_list,
                              ['dec decconv', pde28, pde29, pde1, pde4, pde21, 0,
                               'dec kn unk st', pde15, pde16, pde18, 0,
                               'dec am utils', pde8, pde9, pde10, pde25, 0,
                               'dec sigs', pde6, pde26, pde11, pde27, 0,
                               'dec sr', p0, pde37, pde38, 0,
                               'dec fr', pde11b, pde11, pde12, pde13, pde14, 0,
                               'dec sel', pde30, pde31, pde32, pde33, 0,
                               'dec out class', pde34, pde35, pde36, pde39],
                              [pde21, pde30])
            add_to_vocab_dict(vocab_dict, {pde1: vocab.item,
                                           pde4: vocab.mtr,
                                           pde21: vocab.mtr,
                                           pde11: vocab.pos,
                                           pde11a: vocab.pos,
                                           pde11d: vocab.pos,
                                           pde11e: vocab.pos,
                                           pde12: mem_vocab,
                                           pde13: vocab.pos,
                                           pde14: vocab.item,
                                           pde27: vocab.pos,
                                           pde28: mem_vocab,
                                           pde29: vocab.pos,
                                           pde30: sel_out_vocab,
                                           pde31: vocab.mtr_disp,
                                           pde32: vocab.mtr_disp,
                                           pde33: vocab.mtr_disp})

        if hasattr(model, 'mtr'):
            pmt1 = nengo.Probe(model.mtr.ramp)
            pmt2 = nengo.Probe(model.mtr.ramp_reset_hold)
            pmt2b = nengo.Probe(model.mtr.ramp_sig.init_hold)
            pmt3 = nengo.Probe(model.mtr.motor_stop_input.output)
            pmt4 = nengo.Probe(model.mtr.motor_init.output)
            pmt5 = nengo.Probe(model.mtr.motor_go)
            pmt6 = nengo.Probe(model.mtr.ramp_sig.stop)
            # pmt6 = nengo.Probe(model.mtr.ramp_int_stop)
            pmt7a = nengo.Probe(model.mtr.arm_px_node)
            pmt7b = nengo.Probe(model.mtr.arm_py_node)
            pmt8 = nengo.Probe(model.mtr.pen_down,
                               synapse=0.05)
            pmt11 = nengo.Probe(model.mtr.motor_bypass.output)

            add_to_graph_list(graph_list,
                              ['mtr', p0, pmt1, pmt2, pmt2b, pmt6, pmt3, pmt4, pmt5, pmt11, 0,  # noqa
                               'arm', pmt7a, pmt7b, pmt8])

    return (graph_list[:-1], vocab_dict, [])
