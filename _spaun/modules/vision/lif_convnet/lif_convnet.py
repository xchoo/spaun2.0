import numpy as np

import nengo
from nengo_extras.cuda_convnet import CudaConvnetNetwork


def LIFConvNetVision(vis_data, net=None, net_neuron_type=None):
    if net is None:
        net = nengo.Network(label="LIF Convnet Vision")

    with net:
        input_node = nengo.Node(size_in=vis_data.images_data_dimensions,
                                label='Input')
        input_bias = nengo.Node(output=1)

        # --- LIF vision network proper
        ccnet = CudaConvnetNetwork(vis_data.vision_network_data,
                                   synapse=nengo.synapses.Alpha(0.005))
        nengo.Connection(input_node, ccnet.inputs['data'], synapse=None)

        # Input biases (subtract image mean, add positive bias of 1)
        nengo.Connection(
            input_bias, ccnet.inputs['data'], synapse=None,
            transform=np.ones((vis_data.images_data_dimensions, 1)))
        nengo.Connection(input_bias, ccnet.inputs['data'],
                         transform=-vis_data.images_data_mean.T, synapse=None)

        # --- Gather the ccnet layers for probing
        net.layers = []
        for name in ['conv1_neuron_neurons', 'conv2_neuron_neurons']:
            for ens in ccnet.all_ensembles:
                if ens.label == name:
                    net.layers.append(ens)

        # --- Set up input and outputs to the LIF vision system
        net.input = input_node
        net.raw_output = input_node

        # Output to the visual WM
        net.to_mem_output = nengo.Node(size_in=vis_data.dimensions)
        nengo.Connection(ccnet.layer_outputs[vis_data.mem_okey],
                         net.to_mem_output,
                         transform=vis_data.sps_output_scale, synapse=None)

        # Output to the vision network classifier
        net.to_classify_output = nengo.Node(size_in=vis_data.num_classes)
        # nengo.Connection(ccnet.layer_outputs[vis_data.classify_okey],
        #                  net.to_classify_output[:10],
        #                  transform=1.25 / vis_data.sps_fc10_means,
        #                  synapse=None)
        # nengo.Connection(
        #     ccnet.layer_outputs[vis_data.mem_okey],
        #     net.to_classify_output[10:],
        #     transform=vis_data.sps_output_scale * vis_data.sps[10:, :],
        #     synapse=None)
        nengo.Connection(ccnet.layer_outputs[vis_data.classify_okey],
                         net.to_classify_output,
                         transform=1.0 / vis_data.sps_fc10_means, synapse=None)
    return net
