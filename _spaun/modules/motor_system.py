from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import dec_pos_gate_sp_vecs


class MotorSystem(Module):
    def __init__(self, label="Motor Sys", seed=None, add_to_container=None):
        super(MotorSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        motor_bias = nengo.Node(output=1)

        # Motor init signal
        self.motor_init = cfg.make_thresh_ens(0.75)

        # Motor go signal
        self.motor_go = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(motor_bias, self.motor_go)

        # Motor stop signal
        self.motor_stop_input = cfg.make_thresh_ens()
        nengo.Connection(motor_bias, self.motor_stop_input, synapse=None)
        nengo.Connection(self.motor_stop_input, self.motor_go.neurons,
                         transform=[[-3]] * cfg.n_neurons_ens)

        # Ramp signal generation
        ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
        ramp_reset_thresh = cfg.make_thresh_ens(0.91, radius=1.1)
        self.ramp_reset_hold = cfg.make_thresh_ens(0.07)

        nengo.Connection(self.motor_go, ramp_integrator,
                         transform=cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale)
        nengo.Connection(ramp_integrator, ramp_integrator,
                         synapse=cfg.mtr_ramp_synapse)

        nengo.Connection(ramp_integrator, ramp_reset_thresh)

        nengo.Connection(self.motor_init, self.ramp_reset_hold,
                         transform=1.75, synapse=0.015)
        nengo.Connection(ramp_reset_thresh, self.ramp_reset_hold,
                         transform=1.75, synapse=0.015)
        nengo.Connection(self.ramp_reset_hold, self.ramp_reset_hold,
                         transform=cfg.mtr_ramp_reset_hold_transform)

        nengo.Connection(self.motor_stop_input, ramp_integrator.neurons,
                         transform=[[-3]] * cfg.n_neurons_cconv)
        nengo.Connection(self.ramp_reset_hold, ramp_integrator.neurons,
                         transform=[[-3]] * cfg.n_neurons_cconv)

        ## DEBUG
        self.ramp = ramp_integrator

    def setup_connections(self, parent_net):
        # Set up connections from production system module
        if hasattr(parent_net, 'ps'):
            nengo.Connection(parent_net.ps.task, self.motor_stop_input,
                             transform=[-dec_pos_gate_sp_vecs])
            nengo.Connection(parent_net.ps.task, self.motor_init,
                             transform=[2 * dec_pos_gate_sp_vecs],
                             synapse=0.005)
            nengo.Connection(parent_net.ps.task, self.motor_init,
                             transform=[-3 * dec_pos_gate_sp_vecs],
                             synapse=0.05)
        else:
            warn("MotorSystem Module - Cannot connect from 'ps'")

        # Set up connections from decoding system module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.output_stop,
                             self.motor_stop_input, transform=2)
            nengo.Connection(parent_net.dec.output_stop,
                             self.ramp_reset_hold, transform=1)
            pass
        else:
            warn("MotorSystem Module - Cannot connect from 'dec'")
