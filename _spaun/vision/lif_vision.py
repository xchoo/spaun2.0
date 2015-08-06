import os
import numpy as np
import bisect as bs

import nengo

from .utils import mnist
from .utils import load_image_data

import _spaun

spaun_directory = os.path.split(_spaun.__file__)[0]
vision_filepath = os.path.join(spaun_directory, 'vision')

# --- LIF vision network configurations ---
max_rate = 63.04
intercept = 0.
amp = 1.0 / max_rate
pstc = 0.005

# --- LIF vision network weights configurations ---
vision_network_filename = os.path.join(vision_filepath, 'params.npz')
vision_network_data = np.load(vision_network_filename)
dimensions = vision_network_data['Wc'].shape[0]

weights = vision_network_data['weights']
biases = vision_network_data['biases']
weights_class = vision_network_data['Wc']
biases_class = vision_network_data['bc']

# --- LIF vision network neuron model ---
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
assert np.allclose(neuron_type.gain_bias(np.asarray([max_rate]),
                                         np.asarray([intercept])),
                   (1, 1), atol=1e-2)

# --- Visual associative memory configurations ---
means_filename = os.path.join(vision_filepath, 'class_means.npz')
means_data = np.matrix(1.0 / np.load(means_filename)['means'])

am_threshold = 0.5
am_vis_sps = np.multiply(weights_class.T * amp, means_data.T)

am_num_classes = weights_class.shape[1]

vis_sps_scale = 4.5
# For magic number 4.5, see reference_code/vision_2/data_analysis.py

# --- Mnist data ---
_, _, [images_data, images_labels] = mnist(filepath=vision_filepath)
images_labels = map(str, images_labels)

# --- Spaun symbol data ---
_, _, [symbol_data, symbol_labels] = load_image_data('spaun_sym.pkl.gz',
                                                     filepath=vision_filepath)

# --- Combined image (mnist + spaun symbol) data ---
images_data = np.append(images_data, symbol_data, axis=0)
images_labels = np.append(images_labels, symbol_labels, axis=0)

sorted_labels = np.argsort(images_labels)
images_data = images_data[sorted_labels]
images_labels = images_labels[sorted_labels]

images_data_mean = images_data.mean(axis=0, keepdims=True)
images_data_std = 1.0 / np.maximum(images_data.std(axis=0, keepdims=True),
                                   3e-1)

images_data_dimensions = images_data[0].shape[0]
images_labels_inds = []
images_labels_unique = np.unique(images_labels)
for lbl in images_labels_unique:
    images_labels_inds.append(range(bs.bisect_left(images_labels, lbl),
                                    bs.bisect_right(images_labels, lbl)))


def LIFVision(net=None):
    if net is None:
        net = nengo.Network(label="LIF Vision")

    with net:
        # --- LIF vision network proper
        input_node = nengo.Node(size_in=images_data_dimensions, label='Input')
        input_bias = nengo.Node(output=[1] * images_data_dimensions)

        layers = []
        for i, [W, b] in enumerate(zip(weights, biases)):
            n = b.size
            layer = nengo.Ensemble(n, 1, label='layer %d' % i,
                                   neuron_type=neuron_type,
                                   max_rates=max_rate * np.ones(n),
                                   intercepts=intercept * np.ones(n))
            bias = nengo.Node(output=b)
            nengo.Connection(bias, layer.neurons, transform=np.eye(n),
                             synapse=None)

            if i == 0:
                nengo.Connection(input_node, layer.neurons,
                                 transform=images_data_std * W.T,
                                 synapse=pstc)
                nengo.Connection(input_bias, layer.neurons,
                                 transform=-np.multiply(images_data_mean,
                                                        images_data_std) * W.T,
                                 synapse=pstc)
            else:
                nengo.Connection(layers[-1].neurons, layer.neurons,
                                 transform=W.T * amp, synapse=pstc)

            layers.append(layer)

        # Set up input and outputs to the LIF vision system
        net.input = input_node
        net.output = layers[-1].neurons
        net.raw_output = input_node
    return net
