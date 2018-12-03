import numpy as np

import nengo


def LIFVision(vis_data, stim_data, net=None, net_neuron_type=None):
    if net is None:
        net = nengo.Network(label="LIF Vision")

    if net_neuron_type is None:
        net_neuron_type = vis_data.neuron_type

    with net:
        # --- LIF vision network proper
        input_node = nengo.Node(size_in=stim_data.images_data_dimensions,
                                label='Input')
        input_bias = nengo.Node(output=[1] * stim_data.images_data_dimensions)

        net.layers = []
        for i, [W, b] in enumerate(zip(vis_data.weights, vis_data.biases)):
            n = b.size
            layer = nengo.Ensemble(n, 1, label='layer %d' % i,
                                   neuron_type=net_neuron_type,
                                   max_rates=vis_data.max_rate * np.ones(n),
                                   intercepts=vis_data.intercept * np.ones(n))
            bias = nengo.Node(output=np.array(b))
            nengo.Connection(bias, layer.neurons, transform=np.eye(n),
                             synapse=None)

            if i == 0:
                nengo.Connection(
                    input_node, layer.neurons,
                    transform=stim_data.images_data_std * W.T,
                    synapse=vis_data.pstc)
                nengo.Connection(
                    input_bias, layer.neurons,
                    transform=-np.multiply(stim_data.images_data_mean,
                                           stim_data.images_data_std) * W.T,
                    synapse=vis_data.pstc)
            else:
                nengo.Connection(
                    net.layers[-1].neurons, layer.neurons,
                    transform=W.T * vis_data.amp, synapse=vis_data.pstc)

            net.layers.append(layer)

        # --- Set up input and outputs to the LIF vision system
        net.input = input_node
        net.raw_output = input_node

        # Output to the visual WM
        net.to_mem_output = nengo.Node(size_in=net.layers[-1].n_neurons)
        nengo.Connection(net.layers[-1].neurons, net.to_mem_output,
                         transform=vis_data.sps_output_scale, synapse=None)

        # Output to the vision network classifier
        net.to_classify_output = net.to_mem_output
    return net
