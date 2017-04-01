import numpy as np

import nengo
from nengo_extras.cuda_convnet import CudaConvnetNetwork


def LIFImagenetVision(vis_data, stim_data, net=None, net_neuron_type=None):
    if net is None:
        net = nengo.Network(label="LIF Convnet Imagenet Vision")

    with net:
        input_node = nengo.Node(size_in=stim_data.images_data_dimensions,
                                label='Input')
        raw_output_node = nengo.Node(size_in=stim_data.images_data_dimensions,
                                     label='Raw (Normalized) Output')
        image_mean = nengo.Node(output=stim_data.images_data_mean)

        # --- LIF vision network proper
        ccnet = CudaConvnetNetwork(vis_data.vision_network_data,
                                   synapse=nengo.synapses.Alpha(0.001))
        nengo.Connection(input_node, ccnet.inputs['data'], synapse=None)

        # Input biases (subtract image mean)
        nengo.Connection(image_mean, ccnet.inputs['data'], synapse=None,
                         transform=-1)

        # --- Gather the ccnet layers for probing
        net.layers = []
        for name in ['conv1_neuron_neurons', 'conv2_neuron_neurons',
                     'conv3_neuron_neurons', 'conv4_neuron_neurons',
                     'conv5_neuron_neurons']:
            for ens in ccnet.all_ensembles:
                if ens.label == name:
                    net.layers.append(ens)

        # --- Set up input and outputs to the LIF vision system
        net.input = input_node

        nengo.Connection(net.input, raw_output_node,
                         transform=1.0 / stim_data.max_pixel_value,
                         synapse=None)
        net.raw_output = raw_output_node

        # Output to the visual WM
        # HACK: Output zeros to VIS WM (to avoid 1000D VIS WM)
        net.to_mem_output = nengo.Node(np.zeros(vis_data.dimensions))

        # Output to the vision network classifier
        net.to_classify_output = \
            nengo.Node(size_in=vis_data.output_dimensions)
        nengo.Connection(
            ccnet.layer_outputs[vis_data.spaun_sym_okey],
            net.to_classify_output[:vis_data.spaun_sym_out_dimensions],
            synapse=None)
        nengo.Connection(
            ccnet.output,
            net.to_classify_output[vis_data.spaun_sym_out_dimensions:],
            synapse=None)
    return net
