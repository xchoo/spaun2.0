import numpy as np

import nengo
from nengo.networks import EnsembleArray as EA
from nengo.utils.network import with_self
from nengo.dists import Choice, Uniform


class DifferenceFunctionEvaluator(nengo.Network):
    def __init__(self, num_func_points, func_value_range=1.0,
                 func_output_dimensions=1, n_neurons=500, label=None,
                 seed=None, add_to_container=None):
        super(DifferenceFunctionEvaluator, self).__init__(label, seed,
                                                          add_to_container)

        intercept_interval = 2.0 / (num_func_points - 1)

        self.func_output_dimensions = func_output_dimensions

        with self:
            bias_node = nengo.Node(1)

            self.func_input = nengo.Node(size_in=1)
            self.func_output = nengo.Node(size_in=func_output_dimensions)

            self.diff_func_pts = []
            self.diff_func_outputs = []
            self.func_gate_eas = []

            func_domain_inhib_ea = EA(25, num_func_points - 1,
                                      encoders=Choice([[-1]]),
                                      intercepts=Uniform(0,
                                                         intercept_interval))

            # Generate inhibit signal based of the function domain input value
            func_domain_inhib_ea.add_output('const', lambda x: 1)
            inhib_trfm = np.array([np.linspace(-1, 1, num_func_points)[:-1] +
                                   intercept_interval / 2.0])
            nengo.Connection(bias_node, func_domain_inhib_ea.input,
                             transform=-1 - inhib_trfm.T)
            nengo.Connection(self.func_input,
                             func_domain_inhib_ea.input,
                             transform=2 * np.ones((num_func_points - 1, 1)),
                             synapse=None)

            for n in range(func_output_dimensions):
                func_gate_ea = EA(n_neurons, num_func_points,
                                  radius=func_value_range)

                for i, gate in enumerate(func_gate_ea.all_ensembles[1:]):
                    nengo.Connection(func_domain_inhib_ea.const[i],
                                     gate.neurons,
                                     transform=[[-5]] * gate.n_neurons)

                self.func_gate_eas.append(func_gate_ea)
                self.diff_func_pts.append(func_gate_ea.input)
                self.diff_func_outputs.append(func_gate_ea.output)

                nengo.Connection(func_gate_ea.output, self.func_output[n],
                                 transform=np.ones((1, num_func_points)),
                                 synapse=None)

    @with_self
    def make_inhibitable(self, inhibit_scale=3.0):
        self.inhibit = nengo.Node(size_in=1)

        for n in range(self.func_output_dimensions):
            for gate in self.func_gate_eas[n].all_ensembles:
                nengo.Connection(self.inhibit, gate.neurons,
                                 transform=[[-5]] * gate.n_neurons,
                                 synapse=None)


def convert_func_2_diff_func(func_points):
    func_points = np.array(func_points)

    if len(func_points.shape) <= 1:
        func_points[1:] = func_points[1:] - func_points[:-1]
    else:
        func_points[:, 1:] = func_points[:, 1:] - func_points[:, :-1]
    return func_points
