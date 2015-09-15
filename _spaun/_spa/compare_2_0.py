import numpy as np

import nengo
from nengo.dists import Choice, Uniform
from nengo.networks import Product
from nengo.spa.module import Module

from .._networks import VectorNormalize


class Compare(Module):
    """A module for computing the dot product of two inputs.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared
    vocab : Vocabulary, optional
        The vocabulary to use to interpret the vectors
    neurons_per_multiply : int
        Number of neurons to use in each product computation
    """
    def __init__(self, vocab=None, n_neurons_prod=200,
                 dot_product_input_magnitude=1, output_no_match=False,
                 threshold_outputs=False, n_neurons_threshold_ens=50,
                 label=None, seed=None, add_to_container=None):
        super(Compare, self).__init__(label, seed, add_to_container)

        self.input_conns = []
        self.dimensions = vocab.dimensions

        if threshold_outputs:
            if isinstance(threshold_outputs, bool):
                threshold = 0.5
            else:
                threshold = threshold_outputs

        with self:
            self.inputA = nengo.Node(size_in=self.dimensions, label='inputA')
            self.inputB = nengo.Node(size_in=self.dimensions, label='inputB')
            self.output = nengo.Node(size_in=self.dimensions, label='output')

            self.compare = Product(n_neurons_prod, self.dimensions,
                                   input_magnitude=dot_product_input_magnitude)

            connA = nengo.Connection(self.inputA, self.compare.A, synapse=None)
            connB = nengo.Connection(self.inputB, self.compare.B, synapse=None)
            self.input_conns.append(connA)
            self.input_conns.append(connB)

            # DEBUG
            self.dot_prod = nengo.Node(size_in=1)
            nengo.Connection(self.compare.output, self.dot_prod,
                             transform=np.ones((1, self.dimensions)))

            match_vec = vocab.parse('MATCH').v
            if threshold_outputs:
                self.thresh = nengo.Ensemble(n_neurons_threshold_ens,
                                             1, encoders=Choice([[1]]),
                                             intercepts=Uniform(threshold, 1),
                                             eval_points=Uniform(threshold, 1))
                nengo.Connection(self.compare.output, self.thresh,
                                 transform=np.ones((1, self.dimensions)))
                nengo.Connection(self.thresh, self.output,
                                 transform=np.array([match_vec]).T)
            else:
                match_transform = np.array([match_vec] * self.dimensions)
                nengo.Connection(self.compare.output, self.output,
                                 transform=match_transform.T)
            # else:
            #     self.output = nengo.Node(size_in=1, label='output')

            # nengo.Connection(self.compare.output, self.output,
            #                  transform=np.ones((1, dimensions)))

        self.inputs = dict(A=(self.inputA, vocab), B=(self.inputB, vocab))
        self.outputs = dict(default=(self.output, vocab))

        if output_no_match:
            with self:
                no_match_vec = vocab.parse('NO_MATCH').v
                bias_node = nengo.Node(1)
                no_match_thresh = max(1 - threshold, 0.1)

                if threshold_outputs:
                    self.conjugate = \
                        nengo.Ensemble(n_neurons_threshold_ens,
                                       1, encoders=Choice([[1]]),
                                       intercepts=Uniform(no_match_thresh, 1),
                                       eval_points=Uniform(no_match_thresh, 1))
                    nengo.Connection(bias_node, self.conjugate)
                    nengo.Connection(self.compare.output, self.conjugate,
                                     transform=-np.ones((1, self.dimensions)))
                    nengo.Connection(self.conjugate, self.output,
                                     transform=np.array([no_match_vec]).T)
                else:
                    no_match_transform = np.array([no_match_vec] *
                                                  self.dimensions)
                    nengo.Connection(bias_node, self.output,
                                     transform=np.array([no_match_vec]).T)
                    nengo.Connection(self.compare.output, self.output,
                                     transform=-no_match_transform.T)

                # else:
                #     self.conjugate = nengo.Node(size_in=1, label='conjugate')

                # nengo.Connection(bias_node, self.conjugate)
                # nengo.Connection(self.compare.output, self.conjugate,
                #                  transform=-np.ones((1, dimensions)))

            # self.outputs['conjugate'] = (self.conjugate, None)

    def add_input_normalization(self, min_input_magnitude, max_input_magnitude,
                                normalized_output_magnitude=1.0,
                                n_neurons_norm=50, n_neurons_norm_sub=50,
                                n_neurons_prod=150):
        for conn in self.input_conns:
            self.connections.remove(conn)

        with self:
            self.norm_net_A = \
                VectorNormalize(min_input_magnitude, max_input_magnitude,
                                self.dimensions, n_neurons_norm=n_neurons_norm,
                                n_neurons_norm_sub=n_neurons_norm_sub,
                                n_neurons_prod=n_neurons_prod)
            self.norm_net_B = \
                VectorNormalize(min_input_magnitude, max_input_magnitude,
                                self.dimensions, n_neurons_norm=n_neurons_norm,
                                n_neurons_norm_sub=n_neurons_norm_sub,
                                n_neurons_prod=n_neurons_prod)

            nengo.Connection(self.inputA, self.norm_net_A.input, synapse=None)
            nengo.Connection(self.inputB, self.norm_net_B.input, synapse=None)

            nengo.Connection(self.norm_net_A.output, self.compare.A,
                             transform=normalized_output_magnitude)
            nengo.Connection(self.norm_net_B.output, self.compare.B,
                             transform=normalized_output_magnitude)
