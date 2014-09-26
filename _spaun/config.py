import nengo
from nengo.utils.distributions import Uniform


class SpaunConfig():
    def __init__(self):
        self.sp_dim = 512
        self.vis_dim = 200
        self.mtr_dim = 50   # DEBUG
        self.max_enum_list_pos = 8

        self.pstc = 0.005
        self.n_neurons_ens = 50
        self.n_neurons_cconv = 150
        self.n_neurons_mb = 50
        self.max_rates = Uniform(100, 200)
        self.neuron_type = nengo.LIF()

        self.present_interval = 0.15
        self.present_blanks = False
        self.sim_dt = 0.001

        self.mb_decay_val = 0.975
        self.mb_fdbk_val = 1.3

        self.use_opencl = False

cfg = SpaunConfig()
