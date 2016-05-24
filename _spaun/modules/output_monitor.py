import numpy as np
from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from ..experimenter import experiment


class SpaunOutputMonitor(Module):
    def __init__(self, label="Monitor", seed=None, add_to_container=None):
        super(SpaunOutputMonitor, self).__init__(label, seed, add_to_container)
        self.init_module()

        self.mtr_exp_updated = False
        self.mtr_write_min = 0.75
        self.mtr_reset_max = 0.25

    @with_self
    def init_module(self):
        if cfg.use_mpi:
            raise RuntimeError('Not Implemented')
        else:
            self.output = \
                nengo.Node(output=self.monitor_node_func,
                           size_in=len(vocab.mtr.keys) + 3,
                           label='Experiment monitor')

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vocab.vis_main))

    def monitor_node_func(self, t, x):
        # Determine what has been written
        write_inds = x[:-2]
        write_out_ind = int(np.sum(np.where(write_inds > 0.5)))

        mtr_ramp = x[-2]
        mtr_pen_down = x[-1]

        if mtr_pen_down > 0.5:
            if mtr_ramp > self.mtr_write_min and not self.mtr_exp_updated:
                experiment.update_output(t, write_out_ind)
                self.mtr_exp_updated = True
            elif mtr_ramp < self.mtr_reset_max:
                self.mtr_exp_updated = False

    def setup_connections(self, parent_net):
        # Set up connections from motor module
        if hasattr(parent_net, 'mtr'):
            nengo.Connection(parent_net.mtr.ramp, self.output[-2],
                             synapse=0.05)
            nengo.Connection(parent_net.mtr.pen_down, self.output[-1],
                             synapse=0.03)
            nengo.Connection(parent_net.mtr.dec_ind, self.output[:-2],
                             synapse=0.05)
        else:
            warn("Monitor Module - Cannot connect from 'mtr'")
