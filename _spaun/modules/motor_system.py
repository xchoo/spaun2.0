from warnings import warn

import numpy as np
import importlib

import nengo
from nengo.dists import Exponential
from nengo.spa.module import Module
from nengo.utils.network import with_self

from .._networks import DifferenceFunctionEvaluator as DiffFuncEvaltr
from ..configurator import cfg
from ..vocabulator import vocab
from .motor import Controller, Ramp_Signal_Network, forcefield, mtr_data


class MotorSystem(Module):
    def __init__(self, label="Motor Sys", seed=None, add_to_container=None):
        super(MotorSystem, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(output=1)

        # ---------------------- Inputs and outputs ------------------------- #
        # Motor SP input node
        self.motor_sp_in = nengo.Node(size_in=vocab.mtr_dim)

        # Motor bypass signal (runs the ramp, but doesn't output to the arm)
        self.motor_bypass = cfg.make_thresh_ens_net()

        # Motor init signal
        self.motor_init_vis = cfg.make_thresh_ens_net(0.75)
        self.motor_init_ps_task = cfg.make_thresh_ens_net(0.75)
        self.motor_init_ps_dec = cfg.make_thresh_ens_net(0.75)

        # --------------- MOTOR SIGNALLING SYSTEM (STOP / GO) --------------
        # Motor go signal
        self.motor_go = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(bias_node, self.motor_go)

        # Motor stop signal
        self.motor_stop_input = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, self.motor_stop_input.input, synapse=None)
        nengo.Connection(self.motor_stop_input.output, self.motor_go.neurons,
                         transform=[[-3]] * cfg.n_neurons_ens)

        # --------------- MOTOR SIGNALLING SYSTEM (RAMP SIG) --------------
        self.ramp_sig = Ramp_Signal_Network()

        # Signal used to drive the ramp (the constant input signal)
        nengo.Connection(self.motor_go, self.ramp_sig.ramp,
                         transform=cfg.mtr_ramp_synapse * cfg.mtr_ramp_scale,
                         synapse=cfg.mtr_ramp_synapse)

        # Signal to hold the ramp reset as long as the motor system is still
        # initializing (e.g. arm is still going to INIT target)
        # nengo.Connection(self.motor_init.output, self.ramp_sig.init,
        #                  transform=1.75, synapse=0.015)
        nengo.Connection(self.motor_init_vis.output, self.ramp_sig.init,
                         transform=3, synapse=0.008)
        # nengo.Connection(self.motor_init_vis.output, self.ramp_sig.init,
        #                  transform=-6, synapse=0.05)
        nengo.Connection(self.motor_init_ps_task.output, self.ramp_sig.init,
                         transform=5, synapse=0.008)
        nengo.Connection(self.motor_init_ps_task.output, self.ramp_sig.init,
                         transform=-6, synapse=0.05)
        nengo.Connection(self.motor_init_ps_dec.output, self.ramp_sig.init,
                         transform=5, synapse=0.008)
        nengo.Connection(self.motor_init_ps_dec.output, self.ramp_sig.init,
                         transform=-6, synapse=0.05)

        # Stop the ramp from starting if the stop command has been given
        nengo.Connection(self.motor_stop_input.output, self.ramp_sig.go,
                         transform=-1)

        # --------------- FUNCTION REPLICATOR SYSTEM --------------
        mtr_func_dim = vocab.mtr_dim // 2
        func_eval_net = DiffFuncEvaltr(mtr_func_dim,
                                       mtr_data.sp_scaling_factor, 2)
        func_eval_net.make_inhibitable(-5)

        nengo.Connection(self.ramp_sig.ramp, func_eval_net.func_input)
        nengo.Connection(self.motor_bypass.output, func_eval_net.inhibit)

        # Motor path x information - Note conversion to difference of points
        # from path of points
        nengo.Connection(self.motor_sp_in[:mtr_func_dim],
                         func_eval_net.diff_func_pts[0])
        nengo.Connection(self.motor_sp_in[:mtr_func_dim - 1],
                         func_eval_net.diff_func_pts[0][1:], transform=-1)
        # Motor path y information - Note conversion to difference of points
        # from path of points
        nengo.Connection(self.motor_sp_in[mtr_func_dim:],
                         func_eval_net.diff_func_pts[1])
        nengo.Connection(self.motor_sp_in[mtr_func_dim:-1],
                         func_eval_net.diff_func_pts[1][1:], transform=-1)

        # --------------- MOTOR ARM CONTROL -----------------
        arm_obj = cfg.mtr_arm_class()

        if arm_obj is not None:
            arm_rest_coord = np.array(arm_obj.position(q=arm_obj.rest_angles,
                                                       ee_only=True))
            # Note: arm_rest_coord is only used for initialization & startup
            #       transients
            arm_node = nengo.Node(output=lambda t, x, dt=cfg.sim_dt,
                                  arm=arm_obj: arm.apply_torque(x, dt),
                                  size_in=arm_obj.DOF)

            kp = mtr_data.kp if cfg.mtr_kp is None else cfg.mtr_kp
            kv1 = mtr_data.kv1 if cfg.mtr_kv1 is None else cfg.mtr_kv1
            kv2 = mtr_data.kv2 if cfg.mtr_kv2 is None else cfg.mtr_kv2

            ctrl_obj = Controller(dt=cfg.sim_dt, arm=arm_obj, kp=kp,
                                  kv=kv1, kv2=kv2, init_target=arm_rest_coord,
                                  arm_class_name=cfg.mtr_arm_type)

            # Make the osc control
            ctrl_net = ctrl_obj.initialize_model()
            self.ctrl_net = ctrl_net

            # Connect output of motor path evaluator to ctrl_net
            nengo.Connection(func_eval_net.func_output, ctrl_net.target,
                             synapse=0.01)

            # Add bias values to the motor path evaluator output (to shift the
            # drawn digit into the drawing box of the arm)
            nengo.Connection(bias_node, ctrl_net.target,
                             transform=[[cfg.mtr_arm_rest_x_bias],
                                        [cfg.mtr_arm_rest_y_bias]],
                             synapse=None)

            # Feed the torque control signal to the arm
            nengo.Connection(ctrl_net.output, arm_node)

            if cfg.mtr_dyn_adaptation:
                # Arm state
                arm_state = nengo.Node(output=lambda t, arm=arm_obj:
                                       np.hstack([arm.q, arm.dq]))

                # Create ensemble for arm dynamics adaptation
                adapt_ens = nengo.Ensemble(cfg.mtr_dyn_adaptation_n_neurons,
                                           arm_obj.DOF * 2)

                # Get information from the arm and connect to learning ensemble
                nengo.Connection(
                    arm_state, adapt_ens,
                    function=lambda x: ctrl_obj.config.CB_scaledown(x))

                # Make the learning connection
                adapt_conn = nengo.Connection(
                    adapt_ens, arm_node, function=lambda x: [0, 0, 0],
                    learning_rule_type=nengo.PES(
                        cfg.mtr_dyn_adaptation_learning_rate),
                    synapse=0.02)
                nengo.Connection(ctrl_net.output, adapt_conn.learning_rule,
                                 transform=-1)

                self.adapt_conn = adapt_conn
                self.adapt_ens = adapt_ens

            # ------ MOTOR ARM FORCE FIELD ADDITION ------
            forcefield_obj = getattr(forcefield, cfg.mtr_forcefield)(arm_obj)
            forcefield_node = nengo.Node(
                size_in=arm_obj.DOF,
                output=lambda t, x: forcefield_obj.generate(x))
            nengo.Connection(ctrl_net.output, forcefield_node, synapse=None)
            nengo.Connection(forcefield_node, arm_node,
                             synapse=cfg.mtr_forcefield_synapse)

            self.ff_node = forcefield_node

            # ------ ARM ZERO-CENTERED END EFFECTOR POSITION ------
            # ## Note: ctrl_net already has an internal node that gets info
            #          from arm_obj (i.e. state information). So an external
            #          connection is not required

            zero_centered_arm_ee_loc = \
                nengo.Node(output=lambda t,
                           bias=np.array([cfg.mtr_arm_rest_x_bias,
                                          cfg.mtr_arm_rest_y_bias]),
                           arm=arm_obj: arm.x - bias, label='Centered Arm EE')

        # ------ MOTOR ARM CONTROL SIGNAL FEEDBACK ------
        # X to target norm calculation
        target_thresh = cfg.mtr_tgt_threshold
        target_diff_norm = \
            nengo.Ensemble(150, 2,
                           intercepts=Exponential(0.09, target_thresh,
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

        target_diff_norm_thresh = cfg.make_thresh_ens_net()
        nengo.Connection(target_diff_norm, target_diff_norm_thresh.input,
                         function=lambda x:
                         (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0, transform=2)

        # When the target != arm ee position, stop the ramp signal from
        # starting (reaching to start position)
        # nengo.Connection(target_diff_norm, self.ramp_sig.go, transform=-6,
        #                  function=lambda x:
        #                  (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0,
        #                  synapse=0.01)
        nengo.Connection(target_diff_norm_thresh.output, self.ramp_sig.go,
                         transform=-2, synapse=0.01)
        # When the target != arm ee position, stop the ramp signal from
        # stopping (reaching to end position)
        # nengo.Connection(target_diff_norm, self.ramp_sig.end,
        #                  transform=-8,
        #                  function=lambda x:
        #                  (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0,
        #                  synapse=0.01)
        nengo.Connection(target_diff_norm_thresh.output, self.ramp_sig.end,
                         transform=-2, synapse=0.01)

        # Disable the target_diff_norm when motor bypass
        nengo.Connection(self.motor_bypass.output,
                         target_diff_norm.neurons,
                         transform=[[-10.0]] * target_diff_norm.n_neurons)

        # ------ MOTOR PEN DOWN CONTROL ------
        pen_down = cfg.make_thresh_ens_net()

        # Pen is down by default
        nengo.Connection(bias_node, pen_down.input)

        # Cases when the pen should NOT be down
        nengo.Connection(self.ramp_sig.reset_hold, pen_down.input,
                         transform=-1)
        nengo.Connection(self.ramp_sig.stop, pen_down.input,
                         transform=-1)
        nengo.Connection(self.motor_stop_input.output, pen_down.input,
                         transform=-1, synapse=0.05)
        nengo.Connection(self.motor_bypass.output, pen_down.input,
                         transform=-1)

        # Pen down signal feedback to rest of motor system (tells the ramp to
        # keep going, and the ctrl_net to use only kv1)
        nengo.Connection(pen_down.output, self.ramp_sig.go, transform=12)
        if arm_obj is not None:
            nengo.Connection(pen_down.output, ctrl_net.CB2_inhibit)

        # --------------- For external probes ---------------
        self.ramp_int_stop = self.ramp_sig.stop

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
                nengo.Node(output=lambda t, arm=arm_obj: arm.position()[0])
            self.arm_py_node = \
                nengo.Node(output=lambda t, arm=arm_obj: arm.position()[1])
        else:
            self.arm_px_node = nengo.Node(0)
            self.arm_py_node = nengo.Node(0)

        # Arm ee zero_centered location
        self.zero_centered_arm_ee_loc = zero_centered_arm_ee_loc

        # Target ee zero_centered location
        self.zero_centered_tgt_ee_loc = func_eval_net.func_output

        # Pen down status
        self.pen_down = pen_down.output

        # Ramp signal outputs
        self.ramp = self.ramp_sig.ramp
        self.ramp_reset_hold = self.ramp_sig.reset_hold
        self.ramp_50_75 = self.ramp_sig.ramp_50_75

        # Decode index (number) bypass to monitor module
        self.dec_ind = nengo.Node(size_in=len(vocab.mtr.keys) + 1)

        # Target diff norm
        # self.target_diff_norm_out = nengo.Node(size_in=1)
        # nengo.Connection(target_diff_norm, self.target_diff_norm_out,
        #                  transform=-6,
        #                  function=lambda x:
        #                  (np.sqrt(x[0] ** 2 + x[1] ** 2)) > 0,
        #                  synapse=0.01)
        self.target_diff_norm_out = target_diff_norm_thresh.output

        # motor_init signal
        self.motor_init = nengo.Node(size_in=1)
        nengo.Connection(self.motor_init_vis.output, self.motor_init)
        nengo.Connection(self.motor_init_ps_task.output, self.motor_init)
        nengo.Connection(self.motor_init_ps_dec.output, self.motor_init)

        # ################ DEBUG CODE ###################
        self.arm_state = nengo.Node(output=lambda t, arm=arm_obj:
                                    np.hstack([arm.q, arm.dq]))
        self.arm_dq = nengo.Node(output=lambda t, arm=arm_obj: arm.dq)

    def setup_connections(self, parent_net):
        # Set up connections from vision system module
        if hasattr(parent_net, 'vis'):
            # Initialize the motor ramp signal when QM is presented
            mtr_init_vis_sp_vecs = vocab.main.parse('QM').v
            nengo.Connection(parent_net.vis.output, self.motor_init_vis.input,
                             transform=[1.25 * mtr_init_vis_sp_vecs])
        else:
            warn("MotorSystem Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(parent_net, 'ps'):
            # Motor init signal generation - generates a pulse when ps.task
            # changes to DEC vectors. Also generates a pulse when ps.dec
            # changes to FWD or REV.
            mtr_init_task_sp_vecs = vocab.main.parse('DEC + FWD + REV').v

            nengo.Connection(parent_net.ps.task, self.motor_init_ps_task.input,
                             transform=[1.25 * mtr_init_task_sp_vecs])
            nengo.Connection(parent_net.ps.dec, self.motor_init_ps_dec.input,
                             transform=[1.25 * mtr_init_task_sp_vecs])

            # Motor stop signal - stop the motor output when ps.task
            # is not one of the DEC vectors.
            nengo.Connection(parent_net.ps.task, self.motor_stop_input.input,
                             transform=[mtr_init_task_sp_vecs * -1.0])

            # Motor stop signal - stop the motor output when ps.state
            # is not in the LEARN state.
            mtr_learn_sp_vecs = vocab.main.parse('LEARN').v
            nengo.Connection(parent_net.ps.state, self.motor_stop_input.input,
                             transform=[mtr_learn_sp_vecs * -1.0])

            # Motor bypass signal, also disable target_diff_norm
            # calculation
            mtr_bypass_task_sp_vecs = vocab.main.parse('CNT').v
            nengo.Connection(parent_net.ps.dec, self.motor_bypass.input,
                             transform=[mtr_bypass_task_sp_vecs * 1.0])
        else:
            warn("MotorSystem Module - Cannot connect from 'ps'")

        # Set up connections from decoding system module
        if hasattr(parent_net, 'dec'):
            nengo.Connection(parent_net.dec.output,
                             self.motor_sp_in)

            nengo.Connection(parent_net.dec.output_stop,
                             self.motor_stop_input.input, transform=2)

            # Stop the ramp from going
            nengo.Connection(parent_net.dec.output_stop,
                             self.ramp_sig.go, transform=-2)
            # Stop the ramp reset signal from starting up the ramp again
            nengo.Connection(parent_net.dec.output_stop,
                             self.ramp_sig.reset, transform=-2)

            # Decode index bypass connection
            nengo.Connection(parent_net.dec.dec_ind_output,
                             self.dec_ind, synapse=None)
        else:
            warn("MotorSystem Module - Cannot connect from 'dec'")


class MotorSystemDummy(Module):
    def __init__(self, label="Motor Sys", seed=None, add_to_container=None):
        super(MotorSystemDummy, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        pass
