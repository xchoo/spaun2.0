import os
import numpy as np
import bisect as bs

import nengo

from .utils import mnist
from .utils import load_image_data


class VisionDataObject(object):
    def __init__(self):
        self.filepath = os.path.dirname(__file__)

        # --- LIF vision network configurations ---
        self.max_rate = 63.04
        self.intercept = 0.
        self.amp = 1.0 / self.max_rate
        self.pstc = 0.005

        # --- LIF vision network weights configurations ---
        self.vision_network_filename = os.path.join(self.filepath,
                                                    'params.npz')
        self.vision_network_data = np.load(self.vision_network_filename, encoding='bytes')
        self.dimensions = self.vision_network_data['Wc'].shape[0]

        self.weights = self.vision_network_data['weights']
        self.biases = self.vision_network_data['biases']
        weights_class = self.vision_network_data['Wc']
        biases_class = self.vision_network_data['bc']

        # --- LIF vision network neuron model ---
        neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
        assert np.allclose(neuron_type.gain_bias(np.asarray([self.max_rate]),
                                                 np.asarray([self.intercept])),
                           (1, 1), atol=1e-2)

        self.neuron_type = neuron_type

        # --- Visual associative memory configurations ---
        means_filename = os.path.join(self.filepath, 'class_means.npz')
        means_data = np.matrix(1.0 / np.load(means_filename)['means'])

        self.am_threshold = 0.5
        self.sps = np.array(np.multiply(weights_class.T * self.amp,
                                        means_data.T))

        self.num_classes = weights_class.shape[1]

        self.sps_scale = 4.5
        # For magic number 4.5, see reference_code/vision_2/data_analysis.py

        # --- Mnist data ---
        _, _, [images_data, images_labels] = \
            mnist(filepath=self.filepath)
        images_labels = list(map(str, images_labels))

        # --- Spaun symbol data ---
        _, _, [symbol_data, symbol_labels] = \
            load_image_data('spaun_sym.pkl.gz', filepath=self.filepath)

        # --- Combined image (mnist + spaun symbol) data ---
        images_data = np.append(images_data, symbol_data, axis=0)
        images_labels = np.append(images_labels, symbol_labels, axis=0)

        sorted_labels = np.argsort(images_labels)
        images_data = images_data[sorted_labels]
        images_labels = images_labels[sorted_labels]

        self.images_data_mean = images_data.mean(axis=0, keepdims=True)
        self.images_data_std = 1.0 / np.maximum(images_data.std(axis=0,
                                                                keepdims=True),
                                                3e-1)

        self.images_data_dimensions = images_data[0].shape[0]
        self.images_labels_inds = []
        self.images_labels_unique = np.unique(images_labels)
        for lbl in self.images_labels_unique:
            self.images_labels_inds.append(range(bs.bisect_left(images_labels,
                                                                lbl),
                                                 bs.bisect_right(images_labels,
                                                                 lbl)))

        self.images_data = images_data

    def get_image(self, label=None, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        if isinstance(label, tuple):
            label = label[0]

        if isinstance(label, int):
            # Case when 'label' given is really just the image index number
            return (self.images_data[label], label)
        elif label is None:
            # Case where you need just a blank image
            return (np.zeros(self.images_data_dimensions), -1)
        else:
            # All other cases (usually label is a str)
            image_ind = self.get_image_ind(label, rng)
            return (self.images_data[image_ind], image_ind)

    def get_image_label(self, index):
        for label, indicies in enumerate(self.images_labels_inds):
            if index in indicies:
                return label
        return -1

    def get_image_ind(self, label, rng):
        label_ind = np.where(self.images_labels_unique == label)
        if label_ind[0].shape[0] > 0:
            image_ind = rng.choice(
                self.images_labels_inds[label_ind[0][0]])
        else:
            image_ind = rng.choice(len(self.images_labels_inds))
        return image_ind


vis_data = VisionDataObject()
