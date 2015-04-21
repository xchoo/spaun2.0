import numpy as np
from copy import deepcopy as copy
import time

import nengo
from nengo.networks import EnsembleArray
from nengo.synapses import Lowpass
from nengo.dists import Uniform
from nengo.dists import Choice
from nengo.networks import CircularConvolution as CConv

from _spa import MemoryBlock as MB
# from _networks import CircularConvolution as CConv
from _networks import Selector, Router


class SpaunConfig():
    def __init__(self):
        self.seed = int(time.time())
        self.set_seed(self.seed)

        self.sp_dim = 512
        self.vis_dim = 200
        self.mtr_dim = 50   # DEBUG
        self.max_enum_list_pos = 8

        self.pstc = Lowpass(0.005)
        self.n_neurons_ens = 50
        self.n_neurons_cconv = 200
        self.n_neurons_mb = 50
        self.max_rates = Uniform(100, 200)
        self.neuron_type = nengo.LIF()

        self.present_interval = 0.15
        self.present_blanks = False
        self.sim_dt = 0.001

        self.mb_decaybuf_input_scale = 1.75
        self.mb_decay_val = 0.985  # 0.975
        self.mb_fdbk_val = 1.3
        self.mb_config = {'mem_synapse': Lowpass(0.08), 'difference_gain': 6}
        self.mb_gate_scale = 1.2

        self.trans_cconv_radius = 2
        self.trans_ave_scale = 0.2

        self.mtr_ramp_synapse = 0.05
        self.mtr_ramp_reset_hold_transform = 0.94
        self.mtr_ramp_scale = 2
        self.mtr_est_digit_response_time = 1.5 / self.mtr_ramp_scale

        self.dcconv_radius = 2
        self.dcconv_item_in_scale = 0.5

        self.dec_am_min_thresh = 0.20
        self.dec_am_min_diff = 0.1
        self.dec_fr_min_thresh = 0.3
        self.dec_fr_item_in_scale = 0.65

        self._backend = 'ref'
        self.use_opencl = False

        self.probe_data_filename = "probe_data.npz"

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        value = value.lower()
        if value in ['ref']:
            self._backend = 'ref'
        elif value in ['opencl', 'ocl']:
            self._backend = 'ocl'
        elif value in ['mpi', 'bluegene', 'bg']:
            self._backend = 'mpi'
        elif value in ['spinn']:
            self._backend = 'spinn'
        else:
            raise RuntimeError('Exception! "%s" backend is not supported!' %
                               value)

    @property
    def use_ref(self):
        return self._backend == 'ref'

    @property
    def use_opencl(self):
        return self._backend == 'ocl'

    @property
    def use_mpi(self):
        return self._backend == 'mpi'

    @property
    def use_spinn(self):
        return self._backend == 'spinn'

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

    def get_optimal_sp_radius(self, dim=None):
        if dim is None:
            dim = self.sp_dim
        return 3.5 / np.sqrt(dim)

    def get_probe_data_filename(self, label):
        return "_".join(["probe_data",
                         "spc" if self.present_blanks else "nospc",
                         str(type(self.neuron_type).__name__),
                         str(self.sp_dim)]) + \
               ("" if label is None else "_" + label) + \
               ".npz"

    def gen_probe_data_filename(self, label=None):
        self.probe_data_filename = self.get_probe_data_filename(label)

    def make_mem_block(self, **args):
        mb_args = copy(args)
        mb_args['n_neurons'] = \
            args.get('n_neurons', self.n_neurons_mb)
        mb_args['dimensions'] = \
            args.get('dimensions', self.sp_dim)
        mb_args['radius'] = \
            args.get('radius',
                     self.get_optimal_sp_radius(mb_args['dimensions']))
        mb_args['gate_mode'] = \
            args.get('gate_mode', 2)
        for key in self.mb_config.keys():
            mb_args[key] = args.get(key, self.mb_config[key])
        return MB(**mb_args)

    def make_cir_conv(self, **args):
        cconv_args = copy(args)
        cconv_args['n_neurons'] = \
            args.get('n_neurons', self.n_neurons_cconv)
        cconv_args['dimensions'] = \
            args.get('dimensions', self.sp_dim)
        return CConv(**cconv_args)

    def make_ens_array(self, **args):
        ens_args = copy(args)
        ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
        ens_args['n_ensembles'] = args.get('n_ensembles', self.sp_dim)
        ens_args['radius'] = \
            args.get('radius',
                     self.get_optimal_sp_radius(ens_args['n_ensembles']))
        return EnsembleArray(**ens_args)

    def make_thresh_ens(self, threshold=0.5, **args):
        ens_args = copy(args)
        ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
        ens_args['dimensions'] = args.get('dimensions', 1)
        ens_args['intercepts'] = Uniform(threshold, 1)
        ens_args['encoders'] = Choice([[1]])
        ens_args['eval_points'] = Uniform(threshold, 1.1)
        ens_args['n_eval_points'] = 5000
        return nengo.Ensemble(**ens_args)

    def make_selector(self, num_items, gate_gain=3, **args):
        ens_args = copy(args)
        ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
        ens_args['n_ensembles'] = args.get('n_ensembles', self.sp_dim)
        ens_args['radius'] = \
            args.get('radius',
                     self.get_optimal_sp_radius(ens_args['n_ensembles']))
        dimensions = args.pop('dimensions', self.sp_dim)
        return Selector(EnsembleArray, num_items, dimensions, gate_gain,
                        **ens_args)

    def make_router(self, num_items, gate_gain=3, **args):
        ens_args = copy(args)
        ens_args['n_neurons'] = args.get('n_neurons', self.n_neurons_ens)
        ens_args['n_ensembles'] = args.get('n_ensembles', self.sp_dim)
        ens_args['radius'] = \
            args.get('radius',
                     self.get_optimal_sp_radius(ens_args['n_ensembles']))
        dimensions = args.pop('dimensions', self.sp_dim)
        return Router(EnsembleArray, num_items, dimensions, gate_gain,
                      **ens_args)

cfg = SpaunConfig()
