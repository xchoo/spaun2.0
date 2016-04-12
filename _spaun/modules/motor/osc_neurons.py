'''
Copyright (C) 2015 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import importlib

import nengo
import controller


class OSControllerNengo(controller.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, null_control=True, kv2=0, adaptation=None, **kwargs):

        super(OSControllerNengo, self).__init__(**kwargs)

        self.null_control = null_control

        if self.target is None:
            self.target = self.gen_target()

        # self.initialized = False
        self.adaptation = adaptation
        self.block_output = False

        self.kv2 = kv2

    def set_x(self, t, x):
        self.x = np.array(x)
        return self.x

    def set_target(self, t, x):
        self.target = np.array(x)
        return self.target

    def get_target(self, t):
        return self.target

    def get_arm_state(self, t):
        self.arm_state = np.hstack([self.arm.q, self.arm.dq, self.arm.x])
        return self.arm_state

    def initialize_model(self):
        """Generate the Nengo model that will control the arm."""

        config_file = __import__('_spaun.arms.three_link.config',
                                 globals(), locals(), 'OSCConfig')
        config = config_file.OSCConfig(self.adaptation)
        # importlib.import_module('_spaun.arms.three_link.config', 'OSCConfig')
        # from ...arms.three_link.config import OSCConfig
        # config = OSCConfig(self.adaptation)

        # ----------------------------------------------------------------

        model = nengo.Network('OSC', seed=2)
        model.config[nengo.Connection].synapse = nengo.synapses.Lowpass(.001)

        with model:
            # model.config[nengo.Ensemble].neuron_type = nengo.Direct()

            # create input nodes
            arm_node = nengo.Node(self.get_arm_state, size_out=8)

            # def get_target(t):
            #     return model.target
            # model.target = nengo.Node(output=get_target)
            model.target = nengo.Node(output=self.set_target, size_in=2)

            # create neural ensembles
            CB = nengo.Ensemble(**config.CB)
            M1 = nengo.Ensemble(**config.M1)
            M1_mult = nengo.networks.EnsembleArray(**config.M1_mult)

            # create summation / output ensembles
            # u_relay = nengo.Ensemble(n_neurons=1, dimensions=3,
            #                          neuron_type=nengo.Direct())
            u_relay = nengo.Node(size_in=3)
            model.output = u_relay

            def set_output(t, x):
                self.u = x
            output_node = nengo.Node(output=set_output, size_in=3)

            # connect up arm feedback to Cerebellum
            nengo.Connection(arm_node[:6], CB, function=config.CB_scaledown)

            def gen_Mqdq(signal, kv):
                """Generate inertia compensation signal, np.dot(Mq,dq)"""
                # scale things back
                signal = config.CB_scaleup(signal)

                q = signal[:3]
                dq = signal[3:6]

                Mq = self.arm.gen_Mq(q=q)
                # return np.dot(Mq, kv * dq).flatten()
                return np.dot(Mq, self.kv * dq).flatten()

            # connect up Cerebellum inertia compensation to summation node
            nengo.Connection(CB, u_relay,
                             function=lambda x: gen_Mqdq(x, self.kv),
                             transform=-1, synapse=None)  # , synapse=.005)

            model.CB2_inhibit = nengo.Node(size_in=1)
            if self.kv2 != 0:
                CB2 = nengo.Ensemble(**config.CB)
                nengo.Connection(arm_node[:6], CB2,
                                 function=config.CB_scaledown)
                nengo.Connection(CB2, u_relay,
                                 function=lambda x: gen_Mqdq(x, self.kv2),
                                 transform=-1, synapse=None)  # , synapse=.005)
                nengo.Connection(model.CB2_inhibit, CB2.neurons,
                                 transform=([[-config.CB['radius'] * 2.5]] *
                                            config.CB['n_neurons']),
                                 synapse=None)

            # connect up the array for calculating np.dot(JeeTMx, u)
            M1_mult_output = \
                M1_mult.add_output('mult_scaled',
                                   function=config.DP_scaleup_list)

            # connect up control signal input
            for ii in range(0, 6, 2):
                # control is (goal - state) (kp scaling happens on output)
                # connect up goal
                nengo.Connection(model.target[0], M1_mult.input[ii * 2],
                                 transform=1. / config.u_scaling[0])
                nengo.Connection(model.target[1], M1_mult.input[ii * 2 + 2],
                                 transform=1. / config.u_scaling[1])
                # connect up state (-1 on transform)
                nengo.Connection(arm_node[6], M1_mult.input[ii * 2],
                                 transform=-1. / config.u_scaling[0])
                nengo.Connection(arm_node[7], M1_mult.input[ii * 2 + 2],
                                 transform=-1. / config.u_scaling[1])
            # connect up dot product output (post scaling) to summation node
            block_node = \
                nengo.Node(output=lambda t, x: x * (not self.block_output),
                           size_in=3, size_out=3)
            nengo.Connection(M1_mult_output[::2], block_node,
                             transform=self.kp)
            nengo.Connection(M1_mult_output[1::2], block_node,
                             transform=self.kp)
            nengo.Connection(block_node, u_relay)

            # connect up summation node u_relay to arm
            nengo.Connection(u_relay, output_node, synapse=None)

            # connect up arm feedback to M1
            # pass in sin and cos of x[0], x[0]+x[1], x[0]+x[1]+x[2]
            nengo.Connection(arm_node[:3], M1[:6],
                             function=lambda x:
                             config.M1_scaledown(
                             np.hstack([np.sin(np.cumsum(x)),
                                       np.cos(np.cumsum(x))])))

            def gen_JEETMx(signal, use_incorrect_values=False):
                """Generate Jacobian weighted by task-space inertia matrix"""
                # scale things back
                signal = config.M1_scaleup(signal)

                sinq = signal[:3]
                cosq = signal[3:6]

                Mx = self.arm.gen_Mx_sinq_cosq(sinq=sinq, cosq=cosq)
                JEE = self.arm.gen_jacEE_sinq_cosq(sinq=sinq, cosq=cosq,
                          use_incorrect_values=use_incorrect_values) # noqa
                JEETMx = np.dot(JEE.T, Mx)
                return JEETMx.flatten()

            def scaled_gen_JEETMx(signal, **kwargs):
                return config.DP_scaledown(gen_JEETMx(signal, **kwargs))

            if self.adaptation != 'kinematic':
                # set up regular transform connection
                nengo.Connection(M1[:6], M1_mult.input[1::2],
                                 function=scaled_gen_JEETMx, synapse=.005)

            # ------------------ set up null control ------------------
            if self.null_control:
                def gen_null_signal(signal):
                    """Generate the null space control signal"""

                    # calculate our secondary control signal
                    q = config.M1null_scaleup(signal[:3])
                    u_null = (((self.arm.rest_angles - q) + np.pi) %
                              (np.pi * 2) - np.pi)

                    Mq = self.arm.gen_Mq(q=q)
                    JEE = self.arm.gen_jacEE(q=q)
                    Mx = self.arm.gen_Mx(q=q)

                    u_null = np.dot(Mq, self.kp * u_null)

                    # calculate the null space filter
                    Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
                    null_filter = np.eye(3) - np.dot(JEE.T, Jdyn_inv)

                    return np.dot(null_filter, u_null).flatten()

                M1_null = nengo.Ensemble(**config.M1_null)

                nengo.Connection(arm_node[:3], M1_null,
                                 function=config.M1null_scaledown)
                nengo.Connection(M1_null, block_node,
                                 function=gen_null_signal)
            # --------------------------------------------------------
        return model

    def gen_target(self):
        """Generate a random target"""
        gain = np.sum(self.arm.L) * 1.5
        bias = -np.sum(self.arm.L) * .75

        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()
