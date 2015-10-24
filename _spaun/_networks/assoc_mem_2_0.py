import warnings
import numpy as np

import nengo
from nengo.networks import EnsembleArray
from nengo.dists import Uniform, Choice
from nengo.utils.compat import is_iterable
from nengo.utils.network import with_self
from nengo.utils.stdlib import nested

from ..dists import ClippedExpDist

# TODO: Remove bias node for threshold
# TODO: Redo complement for thresholded output


def filtered_step(t, shift=0.5, scale=50, step_val=1):
    return np.maximum(-1 / np.exp((t - shift) * scale) + 1, 0) * step_val


class AssociativeMemory(nengo.Network):
    """Associative memory network.

    Parameters
    ----------
    input_vectors: list or numpy.array
        The list of vectors to be compared against.
    output_vectors: list of numpy.array, optional
        The list of vectors to be produced for each match. If
        not given, the associative memory will act like an auto-associative
        memory (cleanup memory).

    n_neurons: int
        The number of neurons for each of the ensemble (where each ensemble
        represents each item in the input_vectors list)

    threshold: float, list, optional
        The association activation threshold.
    input_scales: float, list, optional
        Scaling factor to apply on each of the input vectors. Note that it
        is possible to scale each vector independently.

    inhibitable: boolean, optional
        Flag to indicate if the entire associative memory module is
        inhibitable (entire thing can be shut off).

    threshold_output: boolean, optional
        Flag to indicate if the output vector value should be thresholded.
        By default, the shape of the output of the associative memory is a
        filtered step function, shaped to reach the 0.8 * maximal output value
        at threshold + 0.1, 0.95 * maximal output value at threshold + 0.2, and
        maximal output value (1) at threshold + 0.3 (when the radius is 1).

    """
    def __init__(self, input_vectors, output_vectors=None,  # noqa: C901
                 n_neurons=50, threshold=0.3, input_scales=1.0,
                 inhibitable=False, inhibit_scale=1.5, label=None, seed=None,
                 add_to_container=None):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

        # Label prefix for all the ensemble labels
        label_prefix = "" if label is None else label + "_"

        # If output vocabulary is not specified, use input vector list
        # (i.e autoassociative memory)
        if output_vectors is None:
            output_vectors = input_vectors

        # Handle different vector list types
        if is_iterable(input_vectors):
            input_vectors = np.matrix(input_vectors)

        if is_iterable(output_vectors):
            output_vectors = np.matrix(output_vectors)

        # Fail if number of input items and number of output items don't
        # match
        if input_vectors.shape[0] != output_vectors.shape[0]:
            raise ValueError(
                'Number of input vectors does not match number of output '
                'vectors. %d != %d'
                % (input_vectors.shape[0], output_vectors.shape[0]))

        # Handle possible different threshold / input_scale values for each
        # element in the associative memory
        if not is_iterable(threshold):
            threshold = np.array([threshold] * input_vectors.shape[0])
        else:
            threshold = np.array(threshold)
        if threshold.shape[0] != input_vectors.shape[0]:
            raise ValueError(
                'Number of threshold values do not match number of input'
                'vectors. Got: %d, expected %d.' %
                (threshold.shape[0], input_vectors.shape[0]))

        # Handle scaling of each input vector
        if not is_iterable(input_scales):
            input_scale = np.matrix([input_scales] * input_vectors.shape[0])
        else:
            input_scale = np.matrix(input_scale)
        if input_scale.shape[1] != input_vectors.shape[0]:
            raise ValueError(
                'Number of input_scale values do not match number of input'
                'vectors. Got: %d, expected %d.' %
                (input_scale.shape[1], input_vectors.shape[0]))

        # Input and output nodes
        N = input_vectors.shape[0]
        self.num_items = N

        # Scaling factor for exponential distribution and filtered step
        # function
        self.exp_scale = 0.15
        filt_scale = 15
        self.filt_step_func = \
            lambda x: filtered_step(x, 0.0, scale=filt_scale)

        # Evaluation points parameters
        self.n_eval_points = 5000

        # Default configuration to use for the ensembles
        am_ens_config = nengo.Config(nengo.Ensemble)
        am_ens_config[nengo.Ensemble].radius = 1
        am_ens_config[nengo.Ensemble].intercepts = \
            ClippedExpDist(self.exp_scale, 0.0, 1.0)
        am_ens_config[nengo.Ensemble].encoders = Choice([[1]])
        am_ens_config[nengo.Ensemble].eval_points = Uniform(0.0, 1.0)
        am_ens_config[nengo.Ensemble].n_eval_points = self.n_eval_points

        # Store output connections (need to redo them if output thresholding
        # is added by the user - see add_threshold_to_output)
        self.out_conns = []

        # Store the inhibitory connections from elem_output to the default
        # vector ensembles (need to redo them if output thresholding
        # is added by the user - see add_threshold_to_output)
        self.default_vector_inhibit_conns = []

        # Flag to indicate if the am network is using thresholded outputs
        self.thresh_ens = None

        # Flag to indicate if the am network is configured with wta
        self._using_wta = False

        # Create the associative memory network
        with nested(self, am_ens_config):
            self.bias_node = nengo.Node(output=1)

            self.input = nengo.Node(size_in=input_vectors.shape[1],
                                    label="input")
            self.output = nengo.Node(size_in=output_vectors.shape[1],
                                     label="output")

            self.elem_input = nengo.Node(size_in=N, label="element input")
            self.elem_output = nengo.Node(size_in=N, label="element output")

            nengo.Connection(self.input, self.elem_input, synapse=None,
                             transform=np.multiply(input_vectors,
                                                   input_scale.T))

            # Make each ensemble
            self.am_ensembles = []
            for i in range(N):
                # Create ensemble
                e = nengo.Ensemble(n_neurons, 1, label=label_prefix + str(i))
                self.am_ensembles.append(e)

                # Connect input and output nodes
                nengo.Connection(self.bias_node, e, transform=-threshold[i],
                                 synapse=None)
                nengo.Connection(self.elem_input[i], e, synapse=None)
                nengo.Connection(e, self.elem_output[i],
                                 function=self.filt_step_func, synapse=None)

            # Configure associative memory to be inhibitable
            if inhibitable:
                # Input node for inhibitory gating signal (if enabled)
                self.inhibit = nengo.Node(size_in=1, label="inhibit")
                nengo.Connection(self.inhibit, self.elem_input,
                                 transform=-np.ones((N, 1)) * inhibit_scale,
                                 synapse=None)
                # Note: We can use decoded connection here because all the
                # encoding vectors are [1]
            else:
                self.inhibit = None

            # Configure utilities output
            self.utilities = self.elem_output

            c = nengo.Connection(self.elem_output, self.output,
                                 transform=output_vectors.T, synapse=None)

            # Add the output connection to the output connection list
            self.out_conns.append(c)

    @with_self
    def add_input(self, name, input_vectors, input_scales=1.0):
        # Handle different vocabulary types
        if is_iterable(input_vectors):
            input_vectors = np.matrix(input_vectors)

        # Handle possible different input_scale values for each
        # element in the associative memory
        if not is_iterable(input_scales):
            input_scales = np.matrix([input_scales] * input_vectors.shape[0])
        else:
            input_scales = np.matrix(input_scales)
        if input_scales.shape[1] != input_vectors.shape[0]:
            raise ValueError(
                'Number of input_scale values do not match number of input '
                'vectors. Got: %d, expected %d.' %
                (input_scales.shape[1], input_vectors.shape[0]))

        input = nengo.Node(size_in=input_vectors.shape[1], label=name)

        if hasattr(self, name):
            raise NameError('Name "%s" already exists as a node in the'
                            'associative memory.')
        else:
            setattr(self, name, input)

        nengo.Connection(input, self.elem_input,
                         synapse=None,
                         transform=np.multiply(input_vectors, input_scales.T))

    @with_self
    def add_output(self, name, output_vectors):
        # Handle different vector list types
        if is_iterable(output_vectors):
            output_vectors = np.matrix(output_vectors)

        output = nengo.Node(size_in=output_vectors.shape[1], label=name)

        if hasattr(self, name):
            raise NameError('Name "%s" already exists as a node in the'
                            'associative memory.')
        else:
            setattr(self, name, output)

        if self.thresh_ens is not None:
            c = nengo.Connection(self.thresh_ens.output, output, synapse=None,
                                 transform=output_vectors.T)
        else:
            c = nengo.Connection(self.elem_output, output, synapse=None,
                                 transform=output_vectors.T)

        # Add the output connection to the output connection list
        self.out_conns.append(c)

    def add_default_output_vector(self, output_vector, output_name='output',
                                  n_neurons=50, min_activation_value=0.5):
        # Default configuration to use for the ensembles
        default_ens_config = nengo.Config(nengo.Ensemble)
        default_ens_config[nengo.Ensemble].radius = 1
        default_ens_config[nengo.Ensemble].intercepts = \
            ClippedExpDist(self.exp_scale, 0.0, 1.0)
        default_ens_config[nengo.Ensemble].encoders = Choice([[1]])
        default_ens_config[nengo.Ensemble].eval_points = Uniform(0.0, 1.0)
        default_ens_config[nengo.Ensemble].n_eval_points = self.n_eval_points

        with nested(self, default_ens_config):
            default_vector_ens = nengo.Ensemble(n_neurons, 1,
                                                label=('Default %s vector' %
                                                       output_name))

            nengo.Connection(self.bias_node, default_vector_ens,
                             synapse=None)

            if self.thresh_ens is not None:
                c = nengo.Connection(self.thresh_ens.output,
                                     default_vector_ens,
                                     transform=(-(1.0 / min_activation_value) *
                                                np.ones((1, self.num_items))))
            else:
                c = nengo.Connection(self.elem_output, default_vector_ens,
                                     transform=(-(1.0 / min_activation_value) *
                                                np.ones((1, self.num_items))))

            self.default_output_utility = default_vector_ens
            self.default_output_thresholded_utility = default_vector_ens

            # Add the output connection to the output connection list
            self.default_vector_inhibit_conns.append(c)

            # Make new output class attribute and connect to it
            output = getattr(self, output_name)
            nengo.Connection(default_vector_ens, output,
                             transform=np.matrix(output_vector).T,
                             synapse=None)

            if self.inhibit is not None:
                nengo.Connection(self.inhibit, default_vector_ens,
                                 transform=-1.0, synapse=None)

    @with_self
    def add_wta_network(self, inhibit_scale=1.0, inhibit_synapse=0.005):
        if not self._using_wta:
            self._using_wta = True

            nengo.Connection(self.elem_output, self.elem_input,
                             synapse=inhibit_synapse,
                             transform=((np.eye(self.num_items) - 1) *
                                        inhibit_scale))
        else:
            warnings.warn('AssociativeMemory network is already configured ' +
                          'with a WTA network. Additional add_wta_network ' +
                          'function calls are ignored.')

    def add_threshold_to_outputs(self, n_neurons=50, inhibit_scale=10):
        if self.thresh_ens is not None:
            warnings.warn('AssociativeMemory network is already configured ' +
                          'with thresholded outputs. Additional ' +
                          'add_threshold_to_output function calls are ' +
                          'ignored.')
            return

        # Default configuration to use for the ensembles
        thresh_ens_config = nengo.Config(nengo.Ensemble)
        thresh_ens_config[nengo.Ensemble].radius = 1
        thresh_ens_config[nengo.Ensemble].intercepts = Uniform(0.5, 1.0)
        thresh_ens_config[nengo.Ensemble].encoders = Choice([[1]])
        thresh_ens_config[nengo.Ensemble].eval_points = Uniform(0.75, 1.1)
        thresh_ens_config[nengo.Ensemble].n_eval_points = self.n_eval_points

        with nested(self, thresh_ens_config):
            self.thresh_bias = EnsembleArray(n_neurons, self.num_items,
                                             label='thresh_bias')
            self.thresh_ens = EnsembleArray(n_neurons, self.num_items,
                                            label='thresh_ens')

            nengo.Connection(self.bias_node, self.thresh_bias.input,
                             transform=np.ones((self.num_items, 1)),
                             synapse=None)
            nengo.Connection(self.bias_node, self.thresh_ens.input,
                             transform=np.ones((self.num_items, 1)),
                             synapse=None)
            nengo.Connection(self.elem_output, self.thresh_bias.input,
                             transform=-inhibit_scale)
            nengo.Connection(self.thresh_bias.output, self.thresh_ens.input,
                             transform=-inhibit_scale)

            self.thresholded_utilities = self.thresh_ens.output

            # Reroute the thresh_ens output to default vector ensembles,
            # and remove the original connections
            conn_list = []
            for conn in self.default_vector_inhibit_conns:
                c = nengo.Connection(self.thresh_ens.output, conn.post,
                                     transform=conn.transform,
                                     synapse=conn.synapse)
                self.connections.remove(conn)
                conn_list.append(c)
            self.default_vector_inhibit_conns = conn_list

            # Reroute the thresh_ens output to the output nodes, and remove the
            # original connections
            conn_list = []
            for conn in self.out_conns:
                c = nengo.Connection(self.thresh_ens.output, conn.post,
                                     transform=conn.transform,
                                     synapse=conn.synapse)
                self.connections.remove(conn)
                conn_list.append(c)
            self.out_conns = conn_list

            # Make inhibitory connection if inhibit option is set
            if self.inhibit is not None:
                for e in self.thresh_ens.ensembles:
                    nengo.Connection(self.inhibit, e,
                                     transform=-1.5, synapse=None)
