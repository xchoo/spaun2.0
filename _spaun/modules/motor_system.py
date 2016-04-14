from warnings import warn

import numpy as np

import nengo
from nengo.dists import Exponential
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import DifferenceFunctionEvaluator as DiffFuncEvaltr
from ..config import cfg
from ..vocabs import mtr_init_task_sp_vecs, mtr_bypass_task_sp_vecs
from ..vocabs import mtr_sp_scale_factor
from .motor import OSController


class MotorSystem(Module):
    def __init__(self, label="Motor Sys", seed=None, add_to_container=None):
        super(MotorSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(output=1)

        # --------------- MOTOR SIGNALLING SYSTEM (STOP / GO) --------------
        # Motor init signal
        self.motor_init = cfg.make_thresh_ens_net(0.75)

        # Motor go signal
        self.motor_go = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(bias_node, self.motor_go)

        # Motor stop signal
        self.motor_stop_input = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, self.motor_stop_input.input, synapse=None)
        nengo.Connection(self.motor_stop_input.output, self.motor_go.neurons,
                         transform=[[-3]] * cfg.n_neurons_ens)

        # Motor bypass signal (runs the ramp, but doesn't output to the arm)
        self.motor_bypass = cfg.make_thresh_ens_net()

        # Motor SP input node
        self.motor_sp_in = nengo.Node(size_in=cfg.mtr_dim)

        # --------------- MOTOR SIGNALLING SYSTEM (RAMP SIG) --------------
        # Ramp reset hold
        self.ramp_reset_hold = \
            cfg.make_thresh_ens_net(0.07, thresh_func=lambda x: x)
        nengo.Connection(self.motor_init.output, self.ramp_reset_hold.input,
                         transform=1.75, synapse=0.015)
        nengo.Connection(self.ramp_reset_hold.output,
                         self.ramp_reset_hold.input)
        nengo.Connection(bias_node, self.ramp_reset_hold.input,
                         transform=-cfg.mtr_ramp_reset_hold_transform)

        # Ramp integrator go signal (stops ramp integrator when <= 0.5)
        ramp_int_go = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, ramp_int_go.input)
        nengo.Connection(self.ramp_reset_hold.output, ramp_int_go.input,
                         transform=-2)
        nengo.Connection(self.motor_stop_input.output, ramp_int_go.input,
                         transform=-1)

        # Ramp integrator stop signal (inverse of ramp integrator go signal)
        ramp_int_stop = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, ramp_int_stop.input)
        nengo.Connection(ramp_int_go.output, ramp_int_stop.input, transform=-2)

        # Ramp integrator
        ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)
        nengo.Connection(self.motor_go, ramp_integrator,
                         transform=cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale,
                         synapse=cfg.mtr_ramp_synapse)
        # -- Note: We could use the ramp_int_go signal here, but it is noisier
        #    than the motor_go signal
        nengo.Connection(ramp_integrator, ramp_integrator,
                         synapse=cfg.mtr_ramp_synapse)

        # Ramp integrator reset circuitry -- Works like the difference gate in
        # the gated integrator
        ramp_int_reset = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(ramp_integrator, ramp_int_reset)
        nengo.Connection(ramp_int_reset, ramp_integrator,
                         transform=-10, synapse=cfg.mtr_ramp_synapse)
        nengo.Connection(ramp_int_go.output, ramp_int_reset.neurons,
                         transform=[[-3]] * cfg.n_neurons_ens)

        # Ramp end reset signal generator -- Signals when the ramp integrator
        # reaches the top of the ramp slope.
        ramp_reset_thresh = cfg.make_thresh_ens_net(0.91, radius=1.1)
        nengo.Connection(ramp_reset_thresh.output, self.ramp_reset_hold.input,
                         transform=2.5, synapse=0.015)

        # Misc ramp threshold outputs
        ramp_75 = cfg.make_thresh_ens_net(0.75)
        self.ramp_50_75 = cfg.make_thresh_ens_net(0.5)
        self.ramp = ramp_integrator

        nengo.Connection(ramp_integrator, ramp_reset_thresh.input)
        nengo.Connection(ramp_integrator, ramp_75.input)
        nengo.Connection(ramp_integrator, self.ramp_50_75.input)
        nengo.Connection(ramp_75.output, self.ramp_50_75.input, transform=-3)

        # --------------- FUNCTION REPLICATOR SYSTEM --------------
        mtr_func_dim = cfg.mtr_dim // 2
        func_eval_net = DiffFuncEvaltr(mtr_func_dim, mtr_sp_scale_factor, 2)
        func_eval_net.make_inhibitable(-5)

        nengo.Connection(ramp_integrator, func_eval_net.func_input)
        nengo.Connection(self.motor_bypass.output, func_eval_net.inhibit)

        # Motor path x information
        nengo.Connection(self.motor_sp_in[:mtr_func_dim],
                         func_eval_net.diff_func_pts[0])
        # Motor path y information
        nengo.Connection(self.motor_sp_in[mtr_func_dim:],
                         func_eval_net.diff_func_pts[1])

        # --------------- MOTOR ARM CONTROL -----------------
        arm_obj = cfg.mtr_arm_class()

        if arm_obj is not None:
            arm_rest_coord = np.array(arm_obj.position(q=arm_obj.rest_angles,
                                                       ee_only=True))
            # Note: arm_rest_coord is only used for initialization & startup
            #       transients
            arm_node = nengo.Node(output=lambda t, x, dt=cfg.sim_dt:
                                  arm_obj.apply_torque(x, dt),
                                  size_in=arm_obj.DOF)

            osc_obj = OSController(dt=cfg.sim_dt, arm=arm_obj, kp=cfg.mtr_kp,
                                   kv=cfg.mtr_kv1, kv2=cfg.mtr_kv2,
                                   init_target=arm_rest_coord)

            # Make the osc control
            osc_net = osc_obj.initialize_model()

            # Connect output of motor path evaluator to osc_net
            nengo.Connection(func_eval_net.func_output, osc_net.target,
                             synapse=0.01)

            # Add bias values to the motor path evaluator output (to shift the
            # drawn digit into the drawing box of the arm)
            nengo.Connection(bias_node, osc_net.target,
                             transform=[[cfg.mtr_arm_rest_x_bias],
                                        [cfg.mtr_arm_rest_y_bias]],
                             synapse=None)

            # Feed the torque control signal to the arm
            nengo.Connection(osc_net.output, arm_node)

            # ## Note: osc_net already has an internal node that gets info
            #          from arm_obj (i.e. state information). So an external
            #          connection is not required

            zero_centered_arm_ee_loc = \
                nengo.Node(output=lambda t,
                           bias=np.array([cfg.mtr_arm_rest_x_bias,
                                          cfg.mtr_arm_rest_y_bias]):
                           arm_obj.x - bias)

        # ------ MOTOR ARM CONTROL SIGNAL FEEDBACK ------
        # X to target norm calculation
        target_thresh = cfg.mtr_tgt_threshold
        target_diff_norm = \
            nengo.Ensemble(150, 2,
                           intercepts=Exponential(0.05, target_thresh,
                                                  target_thresh * 2),
                           radius=target_thresh * 2)

        nengo.Connection(func_eval_net.func_output, target_diff_norm,
                         synapse=0.01)
        if arm_obj is not None:
            nengo.Connection(zero_centered_arm_ee_loc, target_diff_norm,
                             transform=-1, synapse=0.01)
        else:
            nengo.Connection(func_eval_net.func_output, target_diff_norm,
                             synapse=0.01)

        nengo.Connection(target_diff_norm, ramp_int_go.input, transform=-5,
                         function=lambda x:
                         (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0,
                         synapse=0.01)
        nengo.Connection(target_diff_norm, ramp_reset_thresh.input,
                         transform=-5,
                         function=lambda x:
                         (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0,
                         synapse=0.01)

        # ------ MOTOR PEN DOWN CONTROL ------
        pen_down = cfg.make_thresh_ens_net()

        # Pen is down by default
        nengo.Connection(bias_node, pen_down.input)

        # Cases when the pen should NOT be down
        nengo.Connection(self.ramp_reset_hold.output, pen_down.input,
                         transform=-1)
        nengo.Connection(ramp_int_stop.output, pen_down.input,
                         transform=-1)
        nengo.Connection(self.motor_stop_input.output, pen_down.input,
                         transform=-1, synapse=0.05)
        nengo.Connection(self.motor_bypass.output, pen_down.input,
                         transform=-1)

        # Pen down signal feedback to rest of motor system (tells the ramp to
        # not stop going, and the osc_net to use only kv1)
        nengo.Connection(pen_down.output, ramp_int_go.input, transform=8)
        if arm_obj is not None:
            nengo.Connection(pen_down.output, osc_net.CB2_inhibit)

        # --------------- For external probes ---------------
        self.ramp_int_stop = ramp_int_stop

        # Motor target output
        self.mtr_path_func_out = nengo.Node(size_in=2)
        nengo.Connection(func_eval_net.diff_func_outputs[0],
                         self.mtr_path_func_out[0],
                         transform=np.ones((1, mtr_func_dim)))
        nengo.Connection(func_eval_net.diff_func_outputs[1],
                         self.mtr_path_func_out[1],
                         transform=np.ones((1, mtr_func_dim)))

        # Arm segments joint locations
        if arm_obj is not None:
            self.arm_px_node = \
                nengo.Node(output=lambda t: arm_obj.position()[0])
            self.arm_py_node = \
                nengo.Node(output=lambda t: arm_obj.position()[1])
        else:
            self.arm_px_node = nengo.Node(0)
            self.arm_py_node = nengo.Node(0)

        # Arm ee zero_centered location
        self.zero_centered_arm_ee_loc = zero_centered_arm_ee_loc

        # Target ee zero_centered location
        self.zero_centered_tgt_ee_loc = func_eval_net.func_output

        # Pen down status
        self.pen_down = pen_down.output

    def setup_connections(self, parent_net):
        # Set up connections from production system module
        if hasattr(parent_net, 'ps'):
            # Motor init signal generation - generates a pulse when ps.task
            # changes to DEC vectors.
            nengo.Connection(parent_net.ps.task, self.motor_init.input,
                             transform=[2 * mtr_init_task_sp_vecs],
                             synapse=0.008)
            nengo.Connection(parent_net.ps.task, self.motor_init.input,
                             transform=[-3 * mtr_init_task_sp_vecs],
                             synapse=0.05)

            # Motor stop signal - stop the motor output when ps.task
            # is not one of the DEC vectors.
            nengo.Connection(parent_net.ps.task, self.motor_stop_input.input,
                             transform=[-mtr_init_task_sp_vecs])

            # Motor bypass signal
            nengo.Connection(parent_net.ps.task, self.motor_bypass.input,
                             transform=[mtr_bypass_task_sp_vecs])
        else:
            warn("MotorSystem Module - Cannot connect from 'ps'")

        # Set up connections from decoding system module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.output,
                             self.motor_sp_in)

            nengo.Connection(parent_net.dec.output_stop,
                             self.motor_stop_input.input, transform=2)
            nengo.Connection(parent_net.dec.output_stop,
                             self.ramp_reset_hold.input, transform=-2)
        else:
            warn("MotorSystem Module - Cannot connect from 'dec'")
