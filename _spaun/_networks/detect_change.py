import nengo
from nengo.dists import Choice, Uniform


def DetectChange(net=None, dimensions=1, n_neurons=50, diff_scale=0.3,
                 item_magnitude=1, blank_output_value=1.0):
    # item_magnitude: expected magnitude of one element in the input vector

    if net is None:
        net = nengo.Network(label="Detect Change Network")

    with net:
        net.input = nengo.Node(size_in=dimensions)

        #######################################################################
        input_diff = nengo.networks.EnsembleArray(n_neurons, dimensions,
                                                  label='Input Differentiator',
                                                  intercepts=Uniform(0.1, 1))
        input_diff.add_output('abs', lambda x: abs(x))
        nengo.Connection(net.input, input_diff.input, synapse=0.005,
                         transform=1.0 / item_magnitude)
        nengo.Connection(net.input, input_diff.input, synapse=0.020,
                         transform=-1.0 / item_magnitude)

        # ----- Change detection network -----
        change_detect = nengo.Ensemble(n_neurons, 1,
                                       intercepts=Uniform(0.5, 1),
                                       encoders=Choice([[1]]),
                                       eval_points=Uniform(0.5, 1))
        nengo.Connection(input_diff.abs, change_detect, synapse=0.005,
                         transform=[[diff_scale] * dimensions])

        # ----- Item detection network -----
        net.item_detect = nengo.Node(size_in=1)
        item_detect = nengo.Ensemble(n_neurons * 3, 1)
        nengo.Connection(net.input, item_detect, synapse=0.005,
                         transform=[[1.0 / item_magnitude] * dimensions])
        nengo.Connection(item_detect, net.item_detect,
                         function=lambda x: abs(x))

        # ----- Blank detection network -----
        bias_node = nengo.Node(1)
        blank_detect = nengo.Ensemble(n_neurons, 1,
                                      intercepts=Uniform(0.7, 1),
                                      encoders=Choice([[1]]),
                                      eval_points=Uniform(0.7, 1))
        nengo.Connection(bias_node, blank_detect)
        nengo.Connection(item_detect, blank_detect, synapse=0.005,
                         function=lambda x: abs(x), transform=-1 / 1.5)

        # ----- Output node -----
        net.output = nengo.Node(size_in=1)
        nengo.Connection(change_detect, net.output, synapse=0.0035,
                         transform=1.5)
        nengo.Connection(blank_detect, net.output, synapse=0.01,
                         transform=blank_output_value)

        #######################################################################
        net.input_diff = input_diff.output
        net.change_detect = change_detect
        net.blank_detect = blank_detect

    return net
