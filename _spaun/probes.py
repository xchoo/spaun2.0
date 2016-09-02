import os
import numpy as np

import nengo
from nengo.spa import Vocabulary

from configurator import cfg
from .modules.transform_system import TransformationSystemDummy
from .modules.motor import mtr_data


def idstr(p):
    if not isinstance(p, nengo.Probe):
        return str(p)
    else:
        return str(id(p))


class SpaunProbeConfig(object):
    def __init__(self, spaun_model, spaun_vocab, dt, probe_data_dir,
                 probe_data_filename):
        # Probe config version number
        self.version = 5.0

        # File data names and locations
        self.data_dir = probe_data_dir
        self.data_filename = probe_data_filename
        self.config_filename = probe_data_filename[:-4] + '_cfg.npz'

        # Probe config internal objects
        self.graph_list = []
        # Graph list format: The graph list is a list of strings, where each
        # string is either a graph title, probe_id, or next graph tag
        # - graph titles are suffixed with '.t'
        # - probe_ids are suffixed with 'v.' for value probes,
        #                               'v*' for value probes with legends,
        #                               'V.' for value vocabulary probes,
        #                               'V*' for value vocab probes with legend
        #                               's.' for spike probes,
        #                               's*' for spike probes with legends
        #                               'i.' or 'i*' for image probes
        # - next graph tags are '..'

        self.probe_list = []
        self.label_dict = {}
        self.vocab_dict = {}
        self.ncount_dict = {}
        self.image_dict = {}
        self.path_dict = {}
        self.anim_config = []

        # Store a reference to the original self.m (for use in
        # initialize_probes)
        self.m = spaun_model
        self.v = spaun_vocab
        self.dt = dt

        # Initialize the probes (add to the spaun self.m, and fill in the
        # config lists), then write the probe configuration to file
        self.initialize_probes()
        self.write_config_to_file()

    def probe_null(self):
        return '!!'

    def probe_value(self, probed_obj, synapse=0.005, vocab=None, label=None):
        if isinstance(probed_obj, str):
            probe_id = probed_obj[:-2]
        else:
            with self.m:
                probe = nengo.Probe(probed_obj, synapse=synapse)

            probe_id = idstr(probe)
            if probe_id not in self.probe_list:
                self.probe_list.append(probe_id)

        self.label_dict[probe_id] = label

        if vocab is not None:
            self.vocab_dict[probe_id] = vocab
            return probe_id + 'V.'
        else:
            return probe_id + 'v.'

    def probe_spike(self, probed_obj, n_neurons=20, label=None):
        if isinstance(probed_obj, str):
            if probed_obj[-2] == 's':
                probe_id = probed_obj[:-2]
            else:
                raise ValueError('SpaunProbeConfig.probe_spike - Given ' +
                                 'probe id string, but probed object ' +
                                 'is not a spike probe. Confused. Failing.')
        else:
            with self.m:
                probe = nengo.Probe(probed_obj.neurons)

            probe_id = idstr(probe)
            if probe_id not in self.probe_list:
                self.probe_list.append(probe_id)

        self.ncount_dict[probe_id] = n_neurons
        self.label_dict[probe_id] = label
        return probe_id + 's.'

    def probe_image(self, probed_obj, shape, synapse=None, label=None):
        probe_id = self.probe_value(probed_obj, synapse, label=label)[:-2]
        self.image_dict[probe_id] = shape
        return probe_id + 'i.'

    def probe_path(self, probed_path_obj, probed_pen_down_obj=None,
                   synapse=None, path_xlimits=[-1, 1], path_ylimits=[-1, 1],
                   label=None):
        probe_list = []
        probed_path_id = self.probe_value(probed_path_obj, synapse,
                                          label=label)[:-2]
        probe_list.append(probed_path_id)

        if probed_pen_down_obj is not None:
            probed_pen_down_obj = self.probe_value(probed_pen_down_obj,
                                                   synapse, label=label)[:-2]
            probe_list.append(probed_pen_down_obj)

        self.path_dict[probed_path_id] = [path_xlimits, path_ylimits]

        return '.'.join(probe_list) + 'p.'

    def add_graph(self, graph_title, probe_list, label_list=[]):
        if not isinstance(label_list, list):
            label_list = [label_list]
        if not isinstance(probe_list, list):
            probe_list = [probe_list]

        # Add tag for new graph (only if the graph list is not empty)
        if len(self.graph_list) > 0:
            # Add final ng tag to graph list
            self.graph_list.append('..')

        # Add graph title to graph list
        graph_title = str(graph_title)
        if graph_title.replace('.', '').isdigit():
            graph_title += "**"
        self.graph_list.append(str(graph_title))

        # Add list of probes to graph list
        # - Tag probe id's with * for those that need labels
        for probe_id in probe_list:
            if probe_id in label_list:
                probe_id = probe_id[:-1] + '*'
            self.graph_list.append(probe_id)

    def add_animation(self, key, data_func_name, data_func_params,
                      plot_type_name, plot_type_params):
        data_func_param_dict = {}
        for param_name in data_func_params:
            param = data_func_params[param_name]

            # Strip value tags (auto generated by probe_value) from probe id
            # strs
            if isinstance(param, str):
                data_func_param_dict[param_name] = param[:-2]
            else:
                data_func_param_dict[param_name] = param

        self.anim_config.append({'key': key,
                                 'data_func': data_func_name,
                                 'data_func_params': data_func_param_dict,
                                 'plot_type': plot_type_name,
                                 'plot_type_params': plot_type_params})

    def write_config_to_file(self):
        config_data = {'graph_list': self.graph_list, 'sp_dim': self.v.sp_dim,
                       'vocab_dict': self.vocab_dict, 'prim_vocab': self.v,
                       'ncount_dict': self.ncount_dict,
                       'anim_config': self.anim_config,
                       'image_dict': self.image_dict,
                       'path_dict': self.path_dict,
                       'label_dict': self.label_dict,
                       'dt': self.dt, 'version': self.version}

        np.savez_compressed(os.path.join(self.data_dir, self.config_filename),
                            **config_data)

    def write_simdata_to_file(self, sim, experiment):
        # Generic probe data (time and stimulus sequence)
        probe_data = {'trange': sim.trange(),
                      'stim_seq': experiment.stim_seq_list,
                      'present_interval': experiment.present_interval}

        # Sort out the actual probes from sim
        for probe in sim.data.keys():
            if isinstance(probe, nengo.Probe) and \
               idstr(probe) in self.probe_list:
                probe_data[idstr(probe)] = sim.data[probe]
        np.savez_compressed(os.path.join(self.data_dir,
                                         self.data_filename),
                            **probe_data)

    def initialize_probes(self):
        # To be defined by SpaunProbeConfig subclasses
        raise RuntimeError('SpaunProbeConfig - initialize_probes is meant to' +
                           ' be implemented by SpaunProbeConfig subclasses.')


class ProbeCfgVisOnly(SpaunProbeConfig):
    def initialize_probes(self):
        p0 = self.probe_image(self.m.stim.output, synapse=None, shape=(28, 28))

        pvs1 = self.probe_value(self.m.vis.output, vocab=self.v.vis_main)
        pvs2 = self.probe_value(self.m.vis.neg_attention)
        pvs3 = self.probe_value(self.m.vis.am_utilities)

        pvsp1 = self.probe_spike(self.m.vis.vis_net.layers[0])
        pvsp2 = self.probe_spike(self.m.vis.vis_net.layers[1])

        self.add_graph('Vis', [p0, pvs1, pvs2, pvs3], [pvs1, pvs3])
        self.add_graph('Vis Spikes', [p0, pvsp1, pvsp2])


class ProbeCfgAnimDefault(SpaunProbeConfig):
    def initialize_probes(self):
        if hasattr(self.m, 'vis') and hasattr(self.m, 'mtr'):
            # -------------------- VISION STIMULI PROBES ----------------------
            p0 = self.probe_value(self.m.stim.output, synapse=None)

            self.add_animation(key='vis',
                               data_func_name='generic_single',
                               data_func_params={'data': p0},
                               plot_type_name='imshow',
                               plot_type_params={'shape': (28, 28)})

            # --------------------- MOTOR OUTPUT PROBES -----------------------
            pmt1 = self.probe_value(self.m.mtr.pen_down, synapse=0.05)
            pmt2 = self.probe_value(self.m.mtr.zero_centered_arm_ee_loc,
                                    synapse=0.05)
            pmt3 = self.probe_value(self.m.mtr.zero_centered_tgt_ee_loc,
                                    synapse=0.05)
            pmt4a = self.probe_value(self.m.mtr.arm_px_node)
            pmt4b = self.probe_value(self.m.mtr.arm_py_node)

            self.add_animation(
                key='mtr',
                data_func_name='arm_path',
                data_func_params={'ee_path_data': pmt2,
                                  'target_path_data': pmt3,
                                  'pen_status_data': pmt1,
                                  'arm_posx_data': pmt4a,
                                  'arm_posy_data': pmt4b,
                                  'arm_pos_bias': [cfg.mtr_arm_rest_x_bias,
                                                   cfg.mtr_arm_rest_y_bias]},
                plot_type_name='arm_path_plot',
                plot_type_params={'show_tick_labels': True,
                                  'xlim': (-mtr_data.sp_scaling_factor,
                                           mtr_data.sp_scaling_factor),
                                  'ylim': (-mtr_data.sp_scaling_factor,
                                           mtr_data.sp_scaling_factor)})

            # --------------------- ANIMATION CONFIGURATION -------------------
            self.anim_config.append(
                {'subplot_width': 5,
                 'subplot_height': 5,
                 'max_subplot_cols': 4,
                 'generator_func_params': {'t_index_step': 10}})
        else:
            raise RuntimeError('Unable to setup animation probes. Spaun ' +
                               '`vis` and `mtr` modules are required.')


class ProbeCfgDefault(SpaunProbeConfig):
    def initialize_probes(self):
        # ===================== MAKE DISPLAY VOCABS ===========================
        sub_vocab1 = self.v.enum.create_subset(['POS1*ONE', 'POS2*TWO',
                                                'POS3*THR', 'POS4*FOR',
                                                'POS5*FIV'])

        sub_vocab2 = self.v.main.create_subset(['ADD'])
        sub_vocab2.readonly = False
        sub_vocab2.add('N_ADD', self.v.main.parse('~ADD'))
        sub_vocab2.add('ADD*ADD', self.v.main.parse('ADD*ADD'))
        sub_vocab2.add('ADD*ADD*ADD', self.v.main.parse('ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD',
        #                self.v.main.parse('ADD*ADD*ADD*ADD'))
        # sub_vocab2.add('ADD*ADD*ADD*ADD*ADD',
        #                self.v.main.parse('ADD*ADD*ADD*ADD*ADD'))

        sub_vocab3 = self.v.main.create_subset([])
        sub_vocab3.readonly = False
        # sub_vocab3.add('N_POS1*ONE', self.v.main.parse('~(POS1*ONE)'))
        # sub_vocab3.add('N_POS1*TWO', self.v.main.parse('~(POS1*TWO)'))
        # sub_vocab3.add('N_POS1*THR', self.v.main.parse('~(POS1*THR)'))
        # sub_vocab3.add('N_POS1*FOR', self.v.main.parse('~(POS1*FOR)'))
        # sub_vocab3.add('N_POS1*FIV', self.v.main.parse('~(POS1*FIV)'))
        sub_vocab3.add('ADD', self.v.main.parse('ADD'))
        sub_vocab3.add('INC', self.v.main.parse('INC'))

        vocab_seq_list = self.v.main.create_subset([])
        vocab_seq_list.readonly = False
        for sp_str in ['POS1*ONE', 'POS2*TWO', 'POS3*THR', 'POS4*FOR',
                       'POS5*FIV', 'POS6*SIX', 'POS7*SEV', 'POS8*EIG']:
            vocab_seq_list.add(sp_str, self.v.main.parse(sp_str))

        vocab_rpm = self.v.main.create_subset([])
        vocab_rpm.readonly = False
        for i in [1, 3, 8]:
            sp_str = self.v.num_sp_strs[i]
            vocab_rpm.add('A_(P1+P2+P3)*%s' % sp_str,
                          self.v.main.parse('POS1*%s+POS2*%s+POS3*%s' %
                                            (sp_str, sp_str, sp_str)))
            vocab_rpm.add('N_(P1+P2+P3)*%s' % sp_str,
                          self.v.main.parse('~(POS1*%s+POS2*%s+POS3*%s)' %
                                            (sp_str, sp_str, sp_str)))

        vocab_pos1 = self.v.main.create_subset([])
        vocab_pos1.readonly = False
        for sp_str in self.v.num_sp_strs:
            p1_str = 'POS1*%s' % sp_str
            vocab_pos1.add(p1_str, self.v.main.parse(p1_str))

        # ----------- Default vocabs ------------------
        mem_vocab = vocab_seq_list
        # mem_vocab = vocab_pos1
        # vocab_seq_list = vocab_rpm

        # ========================= MAKE PROBES ===============================
        if hasattr(self.m, 'stim'):
            p0 = self.probe_image(self.m.stim.output, synapse=None,
                                  shape=(28, 28), label='Stim')
        else:
            with self.m:
                self.m.null_node = nengo.Node(0)
            p0 = self.probe_value(self.m.null_node)

        if hasattr(self.m, 'vis') and True:
            net = self.m.vis
            pvs1 = self.probe_value(net.output, vocab=self.v.vis_main,
                                    label='Vis Vcb')
            pvs2 = self.probe_value(net.neg_attention, label='Neg Attn')
            pvs3 = self.probe_value(net.vis_classify_utilities,
                                    label='Vis Utils')
            pvs4 = self.probe_value(net.mb_output, label='Vis MB')
            pvs5 = self.probe_value(net.vis_out, label='Vis Out')
            pvs6 = self.probe_value(net.vis_main_mem.output,
                                    vocab=self.v.vis_main)
            pvs6b = self.probe_value(net.vis_main_mem.input,
                                     vocab=self.v.vis_main)
            pvs6g = self.probe_value(net.vis_main_mem.gate)

            pvsdb1 = self.probe_value(net.rmse_node)
            pvsdb2 = self.probe_value(net.cleanup_node)
            pvsdb3 = self.probe_value(net.diff_node)
            # pvsdb4 = self.probe_value(net.vis_main_mem.output,
            #                           vocab=self.v.vis_main)

            self.add_graph('vis', [p0, pvs1, pvs2, pvs3, pvs6, pvs6b, pvs6g])
            self.add_graph('vis net', [pvs4, pvs5])
            self.add_graph('vis dbg',
                           [p0, pvs6g, pvs6b, pvs6, pvsdb2, pvsdb1, pvsdb3])

        # ############ FOR DEBUGGING VIS DETECT SYSTEM ########################
        # if hasattr(self.m, 'vis') and True:
        #     net = self.m.vis
        #     pvsd1 = self.probe_value(net.detect_change_net.input_diff)
        #     pvsd2 = self.probe_value(net.detect_change_net.item_detect)
        #     pvsd3 = self.probe_value(net.detect_change_net.blank_detect)
        #
        #     self.add_graph('vis detect', [p0, pvsd1, pvsd2, pvsd3])

        if hasattr(self.m, 'ps') and True:
            net = self.m.ps
            pps1 = self.probe_value(net.task, vocab=self.v.ps_task,
                                    label='ps task')
            pps2 = self.probe_value(net.state, vocab=self.v.ps_state,
                                    label='ps state')
            pps3 = self.probe_value(net.dec, vocab=self.v.ps_dec,
                                    label='ps dec')

            pps4 = self.probe_value(net.task_mb.mem1.output,
                                    vocab=self.v.ps_task)
            pps5 = self.probe_value(net.task_mb.mem2.output,
                                    vocab=self.v.ps_task)
            pps6 = self.probe_value(net.task_mb.mem1.input, synapse=None,
                                    vocab=self.v.ps_task)
            pps6b = self.probe_value(net.task_init.output)

            pps7 = self.probe_value(net.state_mb.mem1.output,
                                    vocab=self.v.ps_state)
            pps8 = self.probe_value(net.state_mb.mem2.output,
                                    vocab=self.v.ps_state)
            pps9 = self.probe_value(net.state_mb.mem1.input, synapse=None,
                                    vocab=self.v.ps_state)

            pps10 = self.probe_value(net.dec_mb.mem1.output,
                                     vocab=self.v.ps_dec)
            pps11 = self.probe_value(net.dec_mb.mem2.output,
                                     vocab=self.v.ps_dec)
            pps12 = self.probe_value(net.dec_mb.mem1.input, synapse=None,
                                     vocab=self.v.ps_dec)

            pps13 = self.probe_value(net.task_mb.gate)
            pps14 = self.probe_value(net.state_mb.gate)
            pps15 = self.probe_value(net.dec_mb.gate)
            pps13b = self.probe_value(net.task_mb.mem1.gate)
            pps14b = self.probe_value(net.state_mb.mem1.gate)
            pps15b = self.probe_value(net.dec_mb.mem1.gate)

            pps13r = self.probe_value(net.task_mb.reset)
            pps14r = self.probe_value(net.state_mb.reset)
            pps15r = self.probe_value(net.dec_mb.reset)

            pps16 = self.probe_value(net.action, vocab=self.v.ps_action_learn)
            pps17 = self.probe_value(net.action_in,
                                     vocab=self.v.ps_action_learn)

            self.add_graph('ps', [p0, pps1, pps2, pps3], [pps1, pps2, pps3])
            self.add_graph(
                'ps_task',
                [p0, pps6, pps4, pps5, pps6b, pps13, pps13b, pps13r],
                [pps4, pps5, pps6])
            self.add_graph(
                'ps_state', [p0, pps9, pps7, pps8, pps14, pps14b, pps14r])
            self.add_graph(
                'ps_dec', [p0, pps12, pps10, pps11, pps15, pps15b, pps15r])
            self.add_graph('ps_action', [p0, pps17, pps16], [pps16])

        if hasattr(self.m, 'enc') and True:
            net = self.m.enc
            pen1 = self.probe_value(net.pos_inc.gate)
            # pen2 = self.probe_value(net.pos_inc.pos_mb.gateX)
            pen4 = self.probe_value(net.pos_inc.output, vocab=self.v.pos)
            # pen5 = self.probe_value(net.pos_inc.pos_mb.mem1.output,
            #                         vocab=self.v.pos)
            # pen5b = self.probe_value(net.pos_inc.pos_mb.mem2.output,
            #                          vocab=self.v.pos)
            pen6 = self.probe_value(net.pos_inc.reset)
            pen7 = self.probe_value(net.pos_mb_acc.output, vocab=self.v.pos)
            pen7a = self.probe_value(net.pos_mb_acc.input, vocab=self.v.pos)
            # pen8 = self.probe_value(net.pos_output, vocab=self.v.pos)
            pen9 = self.probe_value(net.pos_inc.reverse)
            pen10 = self.probe_value(net.pos_inc.dir_sel.output,
                                     vocab=self.v.pos)
            pen11 = self.probe_value(net.pos_inc_rev_gate_bias.output)

            self.add_graph(
                'enc',
                [p0, pen1, pen11, pen9, pen10, pen4, pen7, pen7a, pen6],
                [pen4])

        if hasattr(self.m, 'mem') and True:
            net = self.m.mem
            pmm1 = self.probe_value(net.mb1, vocab=mem_vocab)
            pmm1a = self.probe_value(net.mb1_net.mb_reh, vocab=mem_vocab)
            pmm1b = self.probe_value(net.mb1_net.mb_dcy, vocab=mem_vocab)
            pmm2 = self.probe_value(net.mb1_net.gate)
            pmm3 = self.probe_value(net.mb1_net.reset)
            pmm4 = self.probe_value(net.mb2, vocab=vocab_seq_list)
            pmm5 = self.probe_value(net.mb2_net.gate)
            pmm6 = self.probe_value(net.mb2_net.reset)
            pmm7 = self.probe_value(net.mb3, vocab=vocab_seq_list)
            pmm8 = self.probe_value(net.mb3_net.gate)
            pmm9 = self.probe_value(net.mb3_net.reset)

            self.add_graph('mb1', [p0, pmm1, pmm1a, pmm1b, pmm2, pmm3])
            self.add_graph('mb2', [p0, pmm4, pmm5, pmm6])
            self.add_graph('mb3', [p0, pmm7, pmm8, pmm9])

        if hasattr(self.m, 'mem') and True:
            net = self.m.mem
            # pmm1i = self.probe_value(net.input, vocab=mem_vocab)
            # pmm1di = self.probe_value(net.data_input, vocab=mem_vocab)
            pmm1ai = self.probe_value(net.mb1_net.mba.mem1.input,
                                      vocab=mem_vocab)
            pmm1bi = self.probe_value(net.mb1_net.mba.mem2.input,
                                      vocab=mem_vocab)
            pmm1sg1 = self.probe_value(net.select_gate.input0)
            pmm1sg2 = self.probe_value(net.select_gate.input1)
            pmm1sg = self.probe_value(net.select_gate.output)
            pmm1g = self.probe_value(net.mb1_net.gate)
            pmm1gx = self.probe_value(net.mb1_net.mba.gateX)
            pmm1gn = self.probe_value(net.mb1_net.mba.gateN)
            pmm1ag = self.probe_value(net.mb1_net.mba.mem1.gate)
            pmm1bg = self.probe_value(net.mb1_net.mba.mem2.gate)
            # pmm1a1 = self.probe_value(net.mb1_net.mba.mem1.output,
            #                           vocab=mem_vocab)
            # pmm1a2 = self.probe_value(net.mb1_net.mba.mem2.output,
            #                           vocab=mem_vocab)
            # pmm1b1 = self.probe_value(net.mb1_net.mbb.mem1.output,
            #                           vocab=mem_vocab)
            # pmm1b2 = self.probe_value(net.mb1_net.mbb.mem2.output,
            #                           vocab=mem_vocab)
            # pmm1br = self.probe_value(net.mb1_net.mbb.mem1.reset)
            # pmm1bg = self.probe_value(net.mb1_net.mbb.mem1.gate)
            pmm1gg1 = self.probe_value(net.gate_in)
            # pmm1gg2 = self.probe_value(net.gate_in_vis_dbg)
            # pmm1gg3 = self.probe_value(net.gate_sig_bias.output)
            # pmm1gg4 = self.probe_value(net.cnt_gate_sig.output)
            # pmm1gg5 = self.probe_value(net.gate_in_neg_att_dbg)
            pmm1gg6 = self.probe_value(net.gate_in_2)
            pmm1gg7 = self.probe_value(net.gate_sel_none)
            pmm1gg8 = self.probe_value(net.gate_sel_node0)
            pmm1gg9 = self.probe_value(net.gate_sel_node1)
            pmm1gg10 = self.probe_value(net.gate_sel_node1_in)

            self.add_graph(
                'mb1 details',
                [pmm1ai, pmm1bi, pmm1sg1, pmm1sg2, pmm1sg, pmm1g, pmm1gx,
                 pmm1gn, pmm1ag, pmm1bg])
            self.add_graph(
                'mb1 details',
                [pmm1gg6, pmm1gg1, pmm1sg1, pmm1sg2, pmm1gg7, pmm1gg8, pmm1gg9,
                 pmm1gg10, pmm1sg, pmm1g])

        if hasattr(self.m, 'mem') and True:
            net = self.m.mem
            pmm10 = self.probe_value(net.mbave_net.input, vocab=sub_vocab2)
            pmm11 = self.probe_value(net.mbave_net.gate)
            pmm12 = self.probe_value(net.mbave_net.reset)
            pmm13 = self.probe_value(net.mbave, vocab=sub_vocab2)
            pmm13a = self.probe_value(net.mbave)

            self.add_graph('mbave', [p0, pmm10, pmm11, pmm12, pmm13, pmm13a],
                           [pmm10])

        if hasattr(self.m, 'trfm') and \
           not isinstance(self.m.trfm, TransformationSystemDummy):
            net = self.m.trfm
            ptf1 = self.probe_value(net.select_in_a.output, vocab=mem_vocab)
            ptf2 = self.probe_value(net.select_in_b.output, vocab=mem_vocab)
            ptf3 = self.probe_value(net.cconv1.output, vocab=vocab_rpm)
            ptf3b = self.probe_value(ptf3)
            # ptf3c = self.probe_value(ptf3)
            # ptf3d = self.probe_value(ptf3)
            ptf4 = self.probe_value(net.output, vocab=mem_vocab)
            # ptf4b = self.probe_value(ptf4)
            ptf5 = self.probe_value(net.compare.output, vocab=self.v.ps_cmp)
            ptf6 = self.probe_value(net.norm_a.output, vocab=sub_vocab1)
            ptf7 = self.probe_value(net.norm_b.output, vocab=sub_vocab1)
            ptf8 = self.probe_value(net.norm_a.input, vocab=sub_vocab1)
            ptf9 = self.probe_value(net.norm_b.input, vocab=sub_vocab1)
            ptf10 = self.probe_value(net.compare.dot_prod)
            ptf11a = self.probe_value(net.cconv1.A, vocab=vocab_rpm)
            ptf11b = self.probe_value(net.cconv1.B, vocab=vocab_rpm)
            ptf12 = self.probe_value(net.select_out.input6, vocab=self.v.pos1)

            self.add_graph(
                'trfm io', [p0, ptf1, ptf2, ptf4, ptf12], [ptf1, ptf4])
            self.add_graph(
                'trfm cc', [p0, pmm11, ptf3, ptf3b, ptf11a, ptf11b])
            self.add_graph(
                'trfm cmp', [ptf5, ptf8, ptf6, ptf9, ptf7, ptf10],
                [ptf6, ptf7])

        if hasattr(self.m, 'trfm') and \
           not isinstance(self.m.trfm, TransformationSystemDummy):
            nt = self.m.trfm
            ptf5 = self.probe_value(nt.am_trfms.pos1_to_pos, vocab=self.v.pos)
            ptf6 = self.probe_value(nt.am_trfms.pos1_to_num, vocab=self.v.item)
            ptf7 = self.probe_value(nt.am_trfms.num_to_pos1, vocab=self.v.pos1)
            ptf8 = self.probe_value(nt.am_trfms.pos_to_pos1, vocab=self.v.pos1)

            self.add_graph('trfm ams', [p0, ptf5, ptf6, ptf7, ptf8],
                           [ptf5, ptf6, ptf7, ptf8])

        if hasattr(self.m, 'bg') and True:
            pbg1 = self.probe_value(self.m.bg.input)
            pbg2 = self.probe_value(self.m.bg.output)

            self.add_graph('bg', [p0, pbg1, pbg2])

        if hasattr(self.m, 'dec') and True:
            net = self.m.dec
            pde1 = self.probe_value(net.item_dcconv, vocab=self.v.item)
            # pde2 = self.probe_value(net.select_am)
            # pde3 = self.probe_value(net.select_vis)
            pde4 = self.probe_value(net.am_out, synapse=0.01, vocab=self.v.mtr)
            # pde5 = self.probe_value(net.vt_out)
            pde6 = self.probe_value(net.pos_mb_gate_sig.output)
            # pde7 = self.probe_value(net.util_diff_neg)
            pde8 = self.probe_value(net.am_utils)
            pde9 = self.probe_value(net.am2_utils)
            pde10 = self.probe_value(net.util_diff)
            pde11 = self.probe_value(net.pos_recall_mb, vocab=self.v.pos)
            # pde11a = self.probe_value(net.pos_recall_mb_in, vocab=self.v.pos)
            pde11b = self.probe_value(net.fr_recall_mb.gate)
            # pde11c = self.probe_value(net.fr_recall_mb.reset)
            # pde11d = self.probe_value(net.fr_recall_mb.mem1.input,
            #                           vocab=self.v.pos)
            # pde11e = self.probe_value(net.fr_recall_mb.mem2.input,
            #                           vocab=self.v.pos)
            pde12 = self.probe_value(net.fr_dcconv.A, vocab=mem_vocab)
            pde13 = self.probe_value(net.fr_dcconv.B, vocab=self.v.pos)
            pde14 = self.probe_value(net.fr_dcconv.output, vocab=self.v.item)
            pde14b = self.probe_value(net.fr_am_out, vocab=self.v.item)

            # pde12 = self.probe_value(net.recall_mb.gateX)
            # pde13 = self.probe_value(net.recall_mb.gateN)
            # pde14a = self.probe_value(net.recall_mb.mem1.input)
            # pde14b = self.probe_value(net.recall_mb.mem1.output)
            # pde14c = self.probe_value(net.recall_mb.mem2.input)
            # pde14d = self.probe_value(net.recall_mb.mem2.output)
            # pde14e = self.probe_value(net.recall_mb.mem1.diff.output)
            # pde14f = self.probe_value(net.recall_mb.reset)
            # pde14g = self.probe_value(net.recall_mb.mem1.gate)
            # pde12 = self.probe_value(net.dec_am_fr.output)
            # pde13 = self.probe_value(net.dec_am.item_output)
            # pde14 = self.probe_value(net.recall_mb.mem1.output)
            pde15 = self.probe_value(net.output_know)
            pde16 = self.probe_value(net.output_unk)
            pde18 = self.probe_value(net.output_stop)
            # pde19 = self.probe_value(net.am_th_utils)
            # pde20 = self.probe_value(net.fr_th_utils)
            pde21 = self.probe_value(net.output, vocab=self.v.mtr)
            # pde22 = self.probe_value(net.dec_am_fr.input)
            # pde23 = self.probe_value(net.am_def_th_utils)
            # pde24 = self.probe_value(net.fr_def_th_utils)
            pde25 = self.probe_value(net.fr_utils)
            pde26 = self.probe_value(net.pos_mb_gate_bias.output)
            pde27 = self.probe_value(net.pos_acc_input, vocab=self.v.pos)
            pde28 = self.probe_value(net.item_dcconv_a, vocab=mem_vocab)
            pde29 = self.probe_value(net.item_dcconv_b, vocab=self.v.pos)

            sel_out_vocab = Vocabulary(5)
            for n in range(5):
                vec = np.zeros(5)
                vec[n] = 1
                sel_out_vocab.add('SEL%d' % n, vec)

            pde30 = self.probe_value(net.sel_signals, vocab=sel_out_vocab)
            pde31 = self.probe_value(net.select_out.input0,
                                     vocab=self.v.mtr_disp)
            pde32 = self.probe_value(net.select_out.input1,
                                     vocab=self.v.mtr_disp)
            pde33 = self.probe_value(net.select_out.input3,
                                     vocab=self.v.mtr_disp)

            pde34 = self.probe_value(net.out_class_sr_y)
            pde35 = self.probe_value(net.out_class_sr_diff)
            pde36 = self.probe_value(net.out_class_sr_n)
            pde39 = self.probe_value(net.output_classify.fr_utils_n)

            pde37 = self.probe_value(net.serial_decode.inhibit)
            pde38 = self.probe_value(net.free_recall_decode.inhibit)

            self.add_graph(
                'dec decconv', [pde28, pde29, pde1, pde4, pde21], [pde21])
            self.add_graph('dec kn unk st', [pde15, pde16, pde18])
            self.add_graph('dec am utils', [pde8, pde9, pde10, pde25])
            self.add_graph('dec sigs', [pde6, pde26, pde11, pde27])
            self.add_graph('dec sr', [p0, pde37, pde38])
            self.add_graph(
                'dec fr', [pde11b, pde11, pde12, pde13, pde14, pde14b])
            self.add_graph('dec sel', [pde30, pde31, pde32, pde33], [pde30])
            self.add_graph('dec out class', [pde34, pde35, pde36, pde39])

        if hasattr(self.m, 'mtr'):
            net = self.m.mtr
            pmt1 = self.probe_value(net.ramp, label='ramp')
            pmt2 = self.probe_value(net.ramp_reset_hold, label='rst_hold')
            pmt2b = self.probe_value(net.ramp_sig.init_hold, label='init_hold')
            pmt3 = self.probe_value(net.motor_stop_input.output,
                                    label='mtr_si')
            pmt4 = self.probe_value(net.motor_init, label='mtr_init')
            pmt5 = self.probe_value(net.motor_go, label='mtr_go')
            pmt6 = self.probe_value(net.ramp_sig.stop, label='rmp_stop')
            pmt6b = self.probe_value(net.ramp_sig.go, label='rmp_go')
            pmt6c = self.probe_value(net.ramp_sig.end, label='rmp_end')
            pmt7a = self.probe_value(net.arm_px_node, label='arm_px')
            pmt7b = self.probe_value(net.arm_py_node, label='arm_py')
            pmt8 = self.probe_value(net.pen_down, synapse=0.05, label='pen_d')
            pmt11 = self.probe_value(net.motor_bypass.output, label='mtr_byp')
            pmt10 = self.probe_value(net.target_diff_norm_out,
                                     label='tgt_diff_norm')

            self.add_graph('mtr', [p0, pmt1, pmt10, pmt3, pmt4, pmt5, pmt11])
            self.add_graph('ramp', [p0, pmt1, pmt2, pmt2b, pmt6b, pmt6, pmt6c,
                                    pmt8])
            self.add_graph('arm', [pmt7a, pmt7b, pmt8])

        if hasattr(self.m, 'instr'):
            net = self.m.instr
            # pins1 = self.probe_value(net.instr_input)
            pins2 = self.probe_value(net.pos_am.output, vocab=self.v.pos,
                                     label='pos out')
            pins2a = self.probe_value(net.pos_am.input, vocab=self.v.pos,
                                      label='pos in')
            pins2b = self.probe_value(net.pos_am.elem_utilities,
                                      label='pos utils')
            pins3 = self.probe_value(net.output, vocab=self.v.pos1,
                                     label='instr out')
            pins4 = self.probe_value(net.instr_pos_cconv.output,
                                     vocab=self.v.pos, label='cconv pos')
            pins5 = self.probe_value(net.vis_input, vocab=self.v.vis_main)
            pins6 = self.probe_value(net.task_input, vocab=self.v.ps_task)

            pins7 = self.probe_value(net.norm_node1)
            pins8 = self.probe_value(net.norm_node2)
            # pins9 = self.probe_value(net.instr_ea.output)

            pins10a = self.probe_value(net.task_node, vocab=self.v.ps_task,
                                       label='task node')
            pins10 = self.probe_value(net.task_output, vocab=self.v.ps_task,
                                      label='task out')
            pins11a = self.probe_value(net.state_node, vocab=self.v.ps_state,
                                       label='state node')
            pins11 = self.probe_value(net.state_output, vocab=self.v.ps_state,
                                      label='state out')
            pins12a = self.probe_value(net.dec_node, vocab=self.v.ps_dec,
                                       label='dec node')
            pins12 = self.probe_value(net.dec_output, vocab=self.v.ps_dec,
                                      label='dec out')

            pins13 = self.probe_value(net.task_gate_sig, label='task sig')
            pins14 = self.probe_value(net.state_gate_sig, label='state sig')
            pins15 = self.probe_value(net.dec_gate_sig, label='dec sig')
            pins16 = self.probe_value(net.data_gate_sig, label='data sig')

            pins17 = self.probe_value(net.pos_inc.num_2_pos_am.output,
                                      vocab=self.v.pos)
            pins18 = self.probe_value(net.pos_inc.input,
                                      vocab=self.v.item_1_index)
            pins19 = self.probe_value(net.pos_instr, vocab=self.v.pos)
            pins20a = self.probe_value(net.pos_inc.input,
                                       vocab=self.v.vis_main, label='pi in')
            pins20 = self.probe_value(net.pos_inc.output, vocab=self.v.pos,
                                      label='pi out')
            pins21 = self.probe_value(net.pos_inc_init.output,
                                      label='pos init')
            pins22 = self.probe_value(net.pos_inc.gate, label='pos gate')
            pins18a = self.probe_value(net.pos_inc.input_debug)

            pins23 = self.probe_value(net.gate_disable.output)
            pins24 = self.probe_value(net.pos_util_output,
                                      vocab=self.v.ps_task)

            pins25 = self.probe_value(net.pos_inc.num_2_pos_am.output,
                                      vocab=self.v.pos, label='pi am out')
            pins25i = self.probe_value(net.pos_inc.num_2_pos_am.inhibit,
                                       label='pi am inh')
            pins26a = self.probe_value(net.pos_inc.pos_mb.mem1.output,
                                       vocab=self.v.pos, label='pi mb1 out')
            pins26b = self.probe_value(net.pos_inc.pos_mb.mem2.output,
                                       vocab=self.v.pos, label='pi mb2 out')
            pins26r = self.probe_value(net.pos_inc.pos_mb.mem2.reset,
                                       label='pi mb2 rst')
            pins26g = self.probe_value(net.pos_inc.pos_mb.gate,
                                       label='pi mb gate')

            self.add_graph(
                'instr',
                [p0, pins5, pins6, pins4, pins2a, pins2b, pins2, pins3])
            self.add_graph(
                'instr2',
                [p0, pins5, pins6, pins7, pins8, pins23, pins13, pins14,
                 pins15, pins16])
            self.add_graph(
                'instr3',
                [p0, pins10a, pins10, pins13, pins11a, pins11, pins14,
                 pins12a, pins12, pins15, pins24])
            self.add_graph(
                'instr4',
                [p0, pins21, pins22, pins18, pins18a, pins17, pins20a, pins20,
                 pins2a, pins2, pins19])
            self.add_graph(
                'instr5',
                [p0, pins25, pins25i, pins26a, pins26b, pins26g, pins26r])


class ProbeCfgDarpaDebug(SpaunProbeConfig):
    def initialize_probes(self):
        if hasattr(self.m, 'stim'):
            p0 = self.probe_image(self.m.stim.output, synapse=None,
                                  shape=(28, 28), label='Vis Input')
        else:
            p0 = self.probe_null()

        vis_vocab = self.v.vis_main.create_subset(['A', 'OPEN', 'CLOSE', 'QM',
                                                   'ZER', 'ONE', 'TWO', 'THR',
                                                   'FOR', 'FIV', 'SIX', 'SEV',
                                                   'EIG', 'NIN'])
        if hasattr(self.m, 'vis'):
            pvs1 = self.probe_value(self.m.vis.output, vocab=vis_vocab,
                                    label='Vis SP')
            # pvs1a = self.probe_image(self.m.vis.vis_net.raw_output,
            #                          shape=(28, 28), label='Vis Raw Input',
            #                          synapse=None)
            # pvs1b = self.probe_value(self.m.vis.vis_net.output,
            #                          vocab=self.v.vis, label='Vis SP 1')
            pvs1c = self.probe_value(self.m.vis.vis_mem.output,
                                     vocab=self.v.vis, label='Vis Mem SP')

            pvsp1 = self.probe_spike(self.m.vis.vis_net.layers[0],
                                     label='Vis Net L1')
            pvsp2 = self.probe_spike(self.m.vis.vis_net.layers[1],
                                     label='Vis Net L2')
        else:
            pvs1 = self.probe_null()
            pvsp1 = self.probe_null()
            pvsp2 = self.probe_null()

        if hasattr(self.m, 'mem'):
            mem_vocab = self.v.main.create_subset([])
            mem_vocab.readonly = False
            for sp_str in ['POS1*FOR', 'POS2*TWO', 'POS3*SEV', 'POS4*FIV']:
                mem_vocab.add(sp_str, self.v.main.parse(sp_str))
            pmm1 = self.probe_value(self.m.mem.mb1, vocab=mem_vocab,
                                    label='Working Mem')
        else:
            pmm1 = self.probe_null()

        if hasattr(self.m, 'mtr'):
            pmtr1 = self.probe_path(
                self.m.mtr.zero_centered_arm_ee_loc,
                self.m.mtr.pen_down, synapse=0.05,
                path_xlimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                path_ylimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                label='Arm Output')
            pmtr2 = self.probe_value(self.m.mtr.ramp,
                                     label='Mtr Ramp')
            pmtr3 = self.probe_value(self.m.mtr.zero_centered_arm_ee_loc,
                                     label='Arm EE Pos', synapse=None)

            if hasattr(self.m.mtr, 'adapt_conn_debug'):
                pmtr4 = self.probe_value(self.m.mtr.adapt_conn,
                                         label='Adapt Conn', synapse=None)
            else:
                pmtr4 = self.probe_null()

            if hasattr(self.m.mtr, 'ff_node_debug'):
                pmtr5 = self.probe_value(self.m.mtr.ff_node,
                                         label='FF Addition',
                                         synapse=cfg.mtr_forcefield_synapse)
            else:
                pmtr5 = self.probe_null()

            pmtr3b = self.probe_value(self.m.mtr.arm_dq,
                                      label='Arm dq', synapse=None)
            pmtr3c = self.probe_value(self.m.mtr.arm_state,
                                      label='Arm state', synapse=None)
        else:
            pmtr1 = self.probe_null()
            pmtr2 = self.probe_null()
            pmtr3 = self.probe_null()
            pmtr4 = self.probe_null()
            pmtr5 = self.probe_null()
            pmtr3b = self.probe_null()
            pmtr3c = self.probe_null()

        self.add_graph('DEBUG',
                       [p0, pvsp1, pvsp2, pvs1c, pvs1, pmm1,
                        pmtr2, pmtr1, pmtr3, pmtr3b, pmtr3c, pmtr4, pmtr5],
                       [pvs1, pmm1])


class ProbeCfgDarpaVision(SpaunProbeConfig):
    def initialize_probes(self):
        if hasattr(self.m, 'stim'):
            p0 = self.probe_image(self.m.stim.output, synapse=None,
                                  shape=(28, 28), label='Vis Input')
        else:
            p0 = self.probe_null()

        vis_vocab = self.v.vis_main.create_subset(['A', 'OPEN', 'CLOSE', 'QM',
                                                   'ZER', 'ONE', 'TWO', 'THR',
                                                   'FOR', 'FIV', 'SIX', 'SEV',
                                                   'EIG', 'NIN'])
        if hasattr(self.m, 'vis'):
            pvsp1 = self.probe_spike(self.m.vis.vis_net.layers[0],
                                     label='Vis Net L1')
            pvsp2 = self.probe_spike(self.m.vis.vis_net.layers[1],
                                     label='Vis Net L2')

            pvs1 = self.probe_value(self.m.vis.output, vocab=vis_vocab,
                                    label='Vis SP')
        else:
            pvsp1 = self.probe_null()
            pvsp2 = self.probe_null()
            pvs1 = self.probe_null()

        if hasattr(self.m, 'mem'):
            mem_vocab = self.v.main.create_subset([])
            mem_vocab.readonly = False
            for sp_str in ['POS1*FOR', 'POS2*TWO', 'POS3*SEV', 'POS4*FIV']:
                mem_vocab.add(sp_str, self.v.main.parse(sp_str))
            pmm1 = self.probe_value(self.m.mem.mb1, vocab=mem_vocab,
                                    label='Working Mem')
        else:
            pmm1 = self.probe_null()

        if hasattr(self.m, 'mtr'):
            pmtr1 = self.probe_path(
                self.m.mtr.zero_centered_arm_ee_loc,
                self.m.mtr.pen_down, synapse=0.05,
                path_xlimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                path_ylimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                label='Arm Output')
            pmtr2 = self.probe_value(self.m.mtr.ramp,
                                     label='Mtr Ramp')
        else:
            pmtr1 = self.probe_null()
            pmtr2 = self.probe_null()
        self.add_graph('Spiking Vision Network',
                       [p0, pvsp1, pvsp2, pvs1, pmm1, pmtr2, pmtr1],
                       [pvs1, pmm1])


class ProbeCfgDarpaMotor(SpaunProbeConfig):
    def initialize_probes(self):
        if hasattr(self.m, 'stim'):
            p0 = self.probe_image(self.m.stim.output, synapse=None,
                                  shape=(28, 28), label='Vis Input')
        else:
            p0 = self.probe_null()

        if hasattr(self.m, 'mem'):
            mem_vocab = self.v.main.create_subset([])
            mem_vocab.readonly = False
            for sp_str in ['POS1*FOR', 'POS2*TWO', 'POS3*SEV', 'POS4*FIV']:
                mem_vocab.add(sp_str, self.v.main.parse(sp_str))
            pmm1 = self.probe_value(self.m.mem.mb1, vocab=mem_vocab,
                                    label='Working Mem')
        else:
            pmm1 = self.probe_null()

        if hasattr(self.m, 'mtr'):
            pmtr1 = self.probe_spike(self.m.mtr.ctrl_net.CB,
                                     label='Mtr CB')
            pmtr2 = self.probe_spike(self.m.mtr.ctrl_net.M1,
                                     label='Mtr M1')

            pmtr3 = self.probe_value(self.m.mtr.ramp,
                                     label='Mtr Ramp')
            pmtr4 = self.probe_path(
                self.m.mtr.zero_centered_arm_ee_loc,
                self.m.mtr.pen_down, synapse=0.05,
                path_xlimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                path_ylimits=[-mtr_data.sp_scaling_factor * 0.6,
                              mtr_data.sp_scaling_factor * 0.6],
                label='Arm Output')

            pmtr5 = self.probe_spike(self.m.mtr.adapt_ens, label='Adapt Ens')
            pmtr6 = self.probe_value(self.m.mtr.adapt_conn, label='Adapt Ens')
        else:
            pmtr1 = self.probe_null()
            pmtr2 = self.probe_null()
            pmtr3 = self.probe_null()
            pmtr4 = self.probe_null()
            pmtr5 = self.probe_null()
            pmtr6 = self.probe_null()

        self.add_graph('Adaptive Motor Network',
                       [p0, pmm1, pmtr1, pmtr2, pmtr5, pmtr6, pmtr3, pmtr4],
                       [pmm1])
