import numpy as np

import nengo
from nengo.networks import EnsembleArray
from nengo.synapses import Lowpass
from nengo.dists import Uniform, Choice, Exponential
from nengo.networks import CircularConvolution as CConv

from _spa import MemoryBlock as MB
from _spa import SPAEnsembleArray
from _spa.utils import get_optimal_radius
from _networks import AssociativeMemory as AM
from _networks import Selector, Router, VectorNormalize
# from arms import Arm3Link


class SpaunConfig(object):
    def __init__(self):
        self.seed = -1
        self.set_seed(self.seed)

        self.raw_seq_str = ''
        self.raw_seq = None
        self.stim_seq = None

        self.sp_dim = 512
        self.vis_dim = 200
        self.mtr_dim = 50
        self.max_enum_list_pos = 8

        self.num_learn_actions = -1

        self.pstc = Lowpass(0.005)
        self.n_neurons_ens = 50
        self.n_neurons_cconv = 150
        self.n_neurons_mb = 50
        self.n_neurons_am = 50
        self.max_rates = Uniform(200, 400)  # Uniform(100, 200)
        self.neuron_type = nengo.LIF()

        self.present_interval = 0.15
        self.present_blanks = False
        self.sim_dt = 0.001

        self.ps_mb_gain_scale = 2.0
        self.ps_use_am_mb = True

        self.enc_mb_acc_radius_scale = 2.5

        self.mb_rehearsalbuf_input_scale = 1.0  # 1.75
        self.mb_decaybuf_input_scale = 1.5  # 1.75
        self.mb_decay_val = 0.992  # 0.985
        self.mb_fdbk_val = 1.3
        self.mb_config = {'mem_synapse': Lowpass(0.08), 'difference_gain': 6,
                          'gate_gain': 5}
        self.mb_gate_scale = 1.5  # 1.2

        self.trans_cconv_radius = 2
        self.trans_ave_scale = 0.35  # 0.3

        self.dcconv_radius = 2
        self.dcconv_item_in_scale = 0.5

        self.dec_am_min_thresh = 0.35  # 0.20
        self.dec_am_min_diff = 0.1
        self.dec_fr_min_thresh = self.dec_am_min_thresh  # 0.3
        self.dec_fr_item_in_scale = 0.5  # 1.0
        self.dec_fr_to_am_scale = 0.1

        self.mtr_ramp_synapse = 0.05
        self.mtr_ramp_reset_hold_transform = 0.1  # 0.945
        self.mtr_ramp_scale = 2
        self.mtr_est_digit_response_time = 1.0 / self.mtr_ramp_scale + 0.5

        self.mtr_kp = 65
        self.mtr_kv1 = np.sqrt(8)
        self.mtr_kv2 = np.sqrt(18) - self.mtr_kv1
        self.mtr_arm_type = 'three_link'
        self.mtr_arm_rest_x_bias = -0.3
        self.mtr_arm_rest_y_bias = 2.5
        self.mtr_tgt_threshold = 0.075

        self._backend = 'ref'

        self.data_dir = ''
        self.probe_data_filename = 'probe_data.npz'

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, val):
        val = val.lower()
        if val in ['ref']:
            self._backend = 'ref'
        elif val in ['opencl', 'ocl']:
            self._backend = 'ocl'
        elif val in ['mpi', 'bluegene', 'bg']:
            self._backend = 'mpi'
        elif val in ['spinn']:
            self._backend = 'spinn'
        else:
            raise RuntimeError('Exception! "%s" backend is not supported!' %
                               val)

    @property
    def use_ref(self):
        return self.backend == 'ref'

    @property
    def use_opencl(self):
        return self.backend == 'ocl'

    @property
    def use_mpi(self):
        return self.backend == 'mpi'

    @property
    def use_spinn(self):
        return self.backend == 'spinn'

    @property
    def mtr_arm_class(self):
        arm_module = __import__('_spaun.arms.%s' % self.mtr_arm_type,
                                globals(), locals(), 'Arm')
        return arm_module.Arm

    def set_seed(self, seed):
        if seed > 0:
            self.seed = seed
            np.random.seed(self.seed)
            self.rng = np.random.RandomState(self.seed)

    def get_optimal_sp_radius(self, dim=None, subdim=1):
        if dim is None:
            dim = self.sp_dim
        return get_optimal_radius(dim, subdim)

    def get_probe_data_filename(self, label='probe_data', suffix='',
                                ext='npz'):
        suffix = str(suffix).replace('?', '@')

        raw_seq = cfg.raw_seq_str.replace('?', '@').replace(':', ';')
        raw_seq = raw_seq.replace('>', ')').replace('<', '(')

        if self.present_blanks:
            raw_seq = '-'.join(raw_seq)

        return "+".join([label,
                         "_".join([str(type(self.neuron_type).__name__),
                                   str(self.sp_dim)]),
                         raw_seq,
                         str(self.seed)]) + \
               ("" if suffix is '' else '(' + suffix + ')') + "." + ext

    def gen_probe_data_filename(self, label='probe_data', suffix=''):
        self.probe_data_filename = self.get_probe_data_filename(label, suffix)

    def make_assoc_mem(self, input_vectors, output_vectors=None,
                       wta_inhibit_scale=3.5, cleanup_output=True,
                       default_output_vector=None, **args):
        am_args = dict(args)
        am_args['threshold'] = args.get('threshold', 0.5)
        am_args['n_neurons'] = args.get('n_neurons', self.n_neurons_am)

        am_net = AM(input_vectors, output_vectors, **am_args)

        if default_output_vector is not None:
            am_net.add_default_output_vector(default_output_vector)

        if wta_inhibit_scale is not None:
            am_net.add_wta_network(wta_inhibit_scale)

        if cleanup_output:
            am_net.add_cleanup_output(replace_output=True)

        return am_net

    def make_mem_block(self, **args):
        mb_args = dict(args)
        mb_args['n_neurons'] = args.get('n_neurons', self.n_neurons_mb)
        mb_args['dimensions'] = args.get('dimensions', self.sp_dim)
        mb_args['gate_mode'] = args.get('gate_mode', 2)
        mb_args['ens_class'] = args.get('ens_class', SPAEnsembleArray)
        for key in self.mb_config.keys():
            mb_args[key] = args.get(key, self.mb_config[key])
        return MB(**mb_args)

    def make_cir_conv(self, **args):
        cconv_args = dict(args)
        cconv_args['n_neurons'] = args.get('n_neurons', self.n_neurons_cconv)
        cconv_args['dimensions'] = args.get('dimensions', self.sp_dim)
        return CConv(**cconv_args)

    def make_thresh_ens_net(self, threshold=0.5, thresh_func=lambda x: 1,
                            exp_scale=None, net=None, **args):
        if net is None:
            label_str = args.get('label', 'Threshold_Ens_Net')
            net = nengo.Network(label=label_str)
        if exp_scale is None:
            exp_scale = (1 - threshold) / 10.0

        with net:
            ens_args = dict(args)
            ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
            ens_args['dimensions'] = args.get('dimensions', 1)
            ens_args['intercepts'] = \
                Exponential(scale=exp_scale, shift=threshold,
                            high=1)
            ens_args['encoders'] = Choice([[1]])
            ens_args['eval_points'] = Uniform(min(threshold + 0.1, 1.0), 1.1)
            ens_args['n_eval_points'] = 5000

            thresh_ens = nengo.Ensemble(**ens_args)
            net.input = thresh_ens
            net.output = nengo.Node(size_in=1)

            nengo.Connection(thresh_ens, net.output, function=thresh_func,
                             synapse=None)
        return net

    def make_spa_ens_array(self, **args):
        ens_args = dict(args)
        ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
        ens_args['dimensions'] = args.get('dimensions', self.sp_dim)
        return SPAEnsembleArray(**ens_args)

    def make_spa_ens_array_gate(self, threshold_gate=True, **args):
        net = nengo.Network(label='gate')
        with net:
            if threshold_gate:
                thresh_net = self.make_thresh_ens_net()
                net.gate = thresh_net.input
                net.gate_out = thresh_net.output
            else:
                net.gate = nengo.Node(size_in=1)
                net.gate_out = net.gate

            ens_array = self.make_spa_ens_array(**args)
            for ensemble in ens_array.all_ensembles:
                nengo.Connection(net.gate_out, ensemble.neurons,
                                 transform=[[-3]] * ensemble.n_neurons)

            net.input = ens_array.input
            net.output = ens_array.output
        return net

    def make_selector(self, num_items, gate_gain=5, threshold_sel_in=False,
                      **args):
        dimensions = args.pop('dimensions', self.sp_dim)
        ens_class = args.pop('ens_class', SPAEnsembleArray)

        sel_args = dict(args)
        sel_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)

        return Selector(ens_class, num_items, dimensions, gate_gain,
                        threshold_sel_in=threshold_sel_in, **sel_args)

    def make_router(self, num_items, gate_gain=5, threshold_sel_in=False,
                    **args):
        dimensions = args.pop('dimensions', self.sp_dim)
        ens_class = args.pop('ens_class', SPAEnsembleArray)

        rtr_args = dict(args)
        rtr_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)

        return Router(ens_class, num_items, dimensions, gate_gain,
                      threshold_sel_in=threshold_sel_in, **rtr_args)

    def make_norm_net(self, min_input_magnitude=0.7, max_input_magnitude=2.5,
                      **args):
        ens_args = dict(args)
        ens_args['dimensions'] = args.get('dimensions', self.sp_dim)
        ens_args['radius_scale'] = args.get('radius_scale',
                                            self.get_optimal_sp_radius())
        ens_args['n_neurons_norm'] = args.get('n_neurons_norm',
                                              self.n_neurons_ens)
        ens_args['n_neurons_norm_sub'] = args.get('n_neurons_norm_sub',
                                                  self.n_neurons_ens)
        ens_args['n_neurons_prod'] = args.get('n_neurons_prod',
                                              self.n_neurons_cconv)

        norm_net = VectorNormalize(min_input_magnitude, max_input_magnitude,
                                   **ens_args)
        with norm_net:
            norm_net.disable = nengo.Node(size_in=1)
            norm_prod_net = None
            for net in norm_net.networks:
                if net.label == "Product":
                    norm_prod_net = net
            for ensemble in norm_prod_net.all_ensembles:
                nengo.Connection(norm_net.disable, ensemble.neurons,
                                 transform=[[-3]] * ensemble.n_neurons)
        return norm_net

    def make_inhibitable(self, net, inhib_scale=3):
        if hasattr(net, 'inhib'):
            pass
        elif hasattr(net, 'make_inhibitable'):
            net.make_inhibitable(inhib_scale=inhib_scale)
        else:
            with net:
                net.inhib = nengo.Node(size_in=1)
                for e in net.all_ensembles:
                    nengo.Connection(net.inhib, e.neurons,
                                     transform=[[-inhib_scale]] * e.n_neurons,
                                     synapse=None)

cfg = SpaunConfig()
