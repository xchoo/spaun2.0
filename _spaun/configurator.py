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
from _networks import InputGatedMemory as Memory
from _networks import Selector, Router, VectorNormalize
# from arms import Arm3Link

from vocabulator import vocab
from loggerator import logger


class SpaunConfig(object):
    def __init__(self):
        self.seed = -1
        self.set_seed(self.seed)

        self.learn_init_trfm_max = 0.15
        self.learn_init_trfm_bias = 0.05
        self.learn_learning_rate = 1e-4
        self.learn_init_transforms = []

        self.pstc = Lowpass(0.005)
        self.n_neurons_ens = 50
        self.n_neurons_cconv = 150
        self.n_neurons_mb = 50
        self.n_neurons_am = 50
        self.max_rates = Uniform(100, 200)
        self.neuron_type = nengo.LIF()

        self.sim_dt = 0.001

        self.stim_module = 'mnist'
        self.vis_module = 'lif_vision'

        self.vis_detect_dim = 5000

        self.ps_mb_gain_scale = 2.0
        self.ps_mb_gate_scale = 1.25
        self.ps_use_am_mb = True
        self.ps_action_am_threshold = 0.2

        self.enc_mb_acc_radius_scale = 2.5
        self.enc_pos_cleanup_mode = 2

        self.ens_array_subdim = 1

        self.mb_rehearsalbuf_input_scale = 1.0  # 1.75
        self.mb_decaybuf_input_scale = 1.5  # 1.75
        self.mb_decay_val = 0.975
        self.mb_fdbk_val = 1.3
        self.mb_config = {'mem_synapse': Lowpass(0.08), 'difference_gain': 6,
                          'gate_gain': 5}
        self.mb_gate_scale = 1.25  # 1.2
        self.mb_neg_gate_scale = -1.5  # 1.2
        self.mb_neg_attn_scale = 2.0

        self.trans_cconv_radius = 2
        self.trans_ave_scale = 0.25  # 0.3
        self.trans_cmp_threshold = 0.65  # 0.5

        self.dcconv_radius = 2
        self.dcconv_item_in_scale = 0.75  # 0.5

        self.dec_am_min_thresh = 0.30  # 0.20
        self.dec_am_min_diff = 0.1
        self.dec_fr_min_thresh = self.dec_am_min_thresh * 1.2  # 0.3
        self.dec_fr_item_in_scale = 0.65  # 1.0
        self.dec_fr_to_am_scale = 0.25

        self.mtr_ramp_synapse = 0.05
        self.mtr_ramp_reset_hold_transform = 0.1  # 0.945
        self.mtr_ramp_init_hold_transform = 0.01
        self.mtr_ramp_scale = 2
        self.mtr_est_digit_response_time = 1.0 / self.mtr_ramp_scale + 0.60

        self.mtr_module = 'osc'
        self.mtr_kp = None
        self.mtr_kv1 = None
        self.mtr_kv2 = None
        self.mtr_arm_type = 'three_link'
        self.mtr_arm_rest_x_bias = -0.3
        self.mtr_arm_rest_y_bias = 2.5
        self.mtr_tgt_threshold = 0.05  # 0.075

        self.mtr_dyn_adaptation = False
        self.mtr_dyn_adaptation_n_neurons = 1000
        self.mtr_dyn_adaptation_learning_rate = 8e-5
        self.mtr_forcefield = 'NoForcefield'
        self.mtr_forcefield_synapse = 0.0525

        self.instr_cconv_radius = 2.0
        self.instr_out_gain = 1.5
        self.instr_ps_threshold = 0.5
        self.instr_pos_inc_cleanup_mode = 1

        self._backend = 'ref'

        self.data_dir = ''
        self.probe_data_filename = 'probe_data.npz'
        self.probe_graph_config = 'ProbeCfgDefault'
        self.probe_anim_config = 'ProbeCfgAnimDefault'

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
        if self.mtr_arm_type is None:
            return lambda: None

        arm_module = __import__('_spaun.arms.%s' % self.mtr_arm_type,
                                globals(), locals(), 'Arm')
        return arm_module.Arm

    def write_header(self):
        # Write spaun configurator options
        logger.write('# Spaun Configuration Options:\n')
        logger.write('# ----------------------------\n')
        for param_name in sorted(self.__dict__.keys()):
            param_value = getattr(self, param_name)
            if not callable(param_value):
                logger.write('# - %s = %s\n' % (param_name, param_value))
        logger.write('#\n')

    def set_seed(self, seed):
        if seed > 0:
            self.seed = seed
            np.random.seed(self.seed)
            self.rng = np.random.RandomState(self.seed)

    def get_optimal_sp_radius(self, dim=None, subdim=1):
        if dim is None:
            dim = vocab.sp_dim
        return get_optimal_radius(dim, subdim)

    def make_inhibitable(self, net, inhib_scale=3):
        if hasattr(net, 'inhibit'):
            pass
        elif hasattr(net, 'inhib'):
            net.inhibit = net.inhib
        elif hasattr(net, 'make_inhibitable'):
            net.make_inhibitable(inhib_scale=inhib_scale)
        else:
            with net:
                net.inhibit = nengo.Node(size_in=1)
                for e in net.all_ensembles:
                    nengo.Connection(net.inhibit, e.neurons,
                                     transform=[[-inhib_scale]] * e.n_neurons,
                                     synapse=None)

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

    def make_memory(self, **args):
        mem_args = dict(args)
        mem_args['n_neurons'] = args.get('n_neurons', self.n_neurons_mb)
        mem_args['dimensions'] = args.get('dimensions', vocab.sp_dim)
        mem_args['make_ens_func'] = args.get('make_ens_func',
                                             self.make_spa_ens_array)
        for key in self.mb_config.keys():
            mem_args[key] = args.get(key, self.mb_config[key])
        return Memory(**mem_args)

    def make_mem_block(self, **args):
        mb_args = dict(args)
        mb_args['n_neurons'] = args.get('n_neurons', self.n_neurons_mb)
        mb_args['dimensions'] = args.get('dimensions', vocab.sp_dim)
        mb_args['gate_mode'] = args.get('gate_mode', 2)
        mb_args['make_ens_func'] = args.get('make_ens_func',
                                            self.make_spa_ens_array)
        for key in self.mb_config.keys():
            mb_args[key] = args.get(key, self.mb_config[key])
        return MB(**mb_args)

    def make_cir_conv(self, **args):
        cconv_args = dict(args)
        cconv_args['n_neurons'] = args.get('n_neurons', self.n_neurons_cconv)
        cconv_args['dimensions'] = args.get('dimensions', vocab.sp_dim)
        return CConv(**cconv_args)

    def make_thresh_ens_net(self, threshold=0.5, thresh_func=lambda x: 1,
                            exp_scale=None, num_ens=1, net=None, **args):
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

            net.input = nengo.Node(size_in=num_ens)
            net.output = nengo.Node(size_in=num_ens)

            for i in range(num_ens):
                thresh_ens = nengo.Ensemble(**ens_args)
                nengo.Connection(net.input[i], thresh_ens, synapse=None)
                nengo.Connection(thresh_ens, net.output[i],
                                 function=thresh_func, synapse=None)
        return net

    def make_ens_array(self, **args):
        ens_args = dict(args)
        ens_args['radius'] = args.get('radius', 1)
        ens_args['ens_dimensions'] = args.get('ens_dimensions',
                                              self.ens_array_subdim)
        n_ensembles = (ens_args.pop('dimensions', vocab.sp_dim) //
                       ens_args['ens_dimensions'])
        ens_args['n_neurons'] = (args.get('n_neurons', self.n_neurons_ens) *
                                 ens_args['ens_dimensions'])
        ens_args['n_ensembles'] = args.get('n_ensembles', n_ensembles)
        return EnsembleArray(**ens_args)

    def make_spa_ens_array(self, **args):
        ens_args = dict(args)
        ens_args['dimensions'] = args.get('dimensions', vocab.sp_dim)
        ens_dims = ens_args.pop('ens_dimensions', self.ens_array_subdim)
        ens_args['n_neurons'] = (args.get('n_neurons', self.n_neurons_ens) *
                                 ens_dims)
        ens_args['n_ensembles'] = ens_args['dimensions'] // ens_dims

        return SPAEnsembleArray(**ens_args)

    def make_spa_ens_array_gate(self, threshold_gate=True, inhib_scale=3,
                                **args):
        label_str = args.get('label', '')

        net = nengo.Network(label=' '.join([label_str, 'Gate']))
        with net:
            ens_array = self.make_spa_ens_array(**args)
            self.make_inhibitable(ens_array, inhib_scale)

            if threshold_gate:
                thresh_net = self.make_thresh_ens_net()
                net.gate = thresh_net.input

                nengo.Connection(thresh_net.output, ens_array.inhibit)
            else:
                net.gate = ens_array.inhibit

            net.input = ens_array.input
            net.output = ens_array.output
        return net

    def make_selector(self, num_items, gate_gain=5, threshold_sel_in=True,
                      **args):
        dimensions = args.pop('dimensions', vocab.sp_dim)
        make_ens_func = args.pop('make_ens_func', self.make_spa_ens_array)
        n_neurons = args.pop('n_neurons', self.n_neurons_ens)

        sel_args = dict(args)

        return Selector(n_neurons, dimensions, num_items, make_ens_func,
                        gate_gain, threshold_sel_in=threshold_sel_in,
                        **sel_args)

    def make_router(self, num_items, gate_gain=5, threshold_sel_in=True,
                    **args):
        dimensions = args.pop('dimensions', vocab.sp_dim)
        make_ens_func = args.pop('make_ens_func', self.make_spa_ens_array)
        n_neurons = args.pop('n_neurons', self.n_neurons_ens)

        rtr_args = dict(args)

        return Router(n_neurons, dimensions, num_items, make_ens_func,
                      gate_gain, threshold_sel_in=threshold_sel_in,
                      **rtr_args)

    def make_norm_net(self, min_input_magnitude=0.7, max_input_magnitude=2.5,
                      **args):
        ens_args = dict(args)
        ens_args['dimensions'] = args.get('dimensions', vocab.sp_dim)
        ens_args['radius_scale'] = args.get('radius_scale',
                                            self.get_optimal_sp_radius())
        ens_args['n_neurons_norm'] = args.get('n_neurons_norm',
                                              self.n_neurons_ens)
        ens_args['n_neurons_prod'] = args.get('n_neurons_prod',
                                              self.n_neurons_cconv)

        norm_net = VectorNormalize(min_input_magnitude, max_input_magnitude,
                                   **ens_args)
        with norm_net:
            disable_ens = self.make_thresh_ens_net()
            norm_net.disable = disable_ens.input
            for net in norm_net.networks:
                if net.label == "Product":
                    self.make_inhibitable(net, inhib_scale=5.0)
                    nengo.Connection(disable_ens.output, net.inhibit)
        return norm_net

cfg = SpaunConfig()
