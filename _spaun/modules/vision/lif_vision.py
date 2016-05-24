import numpy as np

import nengo

from .data import vis_data as data


def LIFVision(net=None, net_neuron_type=None):
    if net is None:
        net = nengo.Network(label="LIF Vision")

    if net_neuron_type is None:
        net_neuron_type = data.neuron_type

    with net:
        # --- LIF vision network proper
        input_node = nengo.Node(size_in=data.images_data_dimensions,
                                label='Input')
        input_bias = nengo.Node(output=[1] * data.images_data_dimensions)

        layers = []
        for i, [W, b] in enumerate(zip(data.weights, data.biases)):
            n = b.size
            layer = nengo.Ensemble(n, 1, label='layer %d' % i,
                                   neuron_type=net_neuron_type,
                                   max_rates=data.max_rate * np.ones(n),
                                   intercepts=data.intercept * np.ones(n))
            bias = nengo.Node(output=np.array(b))
            nengo.Connection(bias, layer.neurons, transform=np.eye(n),
                             synapse=None)

            if i == 0:
                nengo.Connection(input_node, layer.neurons,
                                 transform=data.images_data_std * W.T,
                                 synapse=data.pstc)
                nengo.Connection(input_bias, layer.neurons,
                                 transform=-np.multiply(data.images_data_mean,
                                                        data.images_data_std) *
                                 W.T,
                                 synapse=data.pstc)
            else:
                nengo.Connection(layers[-1].neurons, layer.neurons,
                                 transform=W.T * data.amp, synapse=data.pstc)

            layers.append(layer)

        # Set up input and outputs to the LIF vision system
        net.input = input_node
        net.output = layers[-1].neurons
        net.raw_output = input_node
    return net
