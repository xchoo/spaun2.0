import os
import numpy as np

import nengo


class LIFVisionDataObject(object):
    def __init__(self):
        self.filepath = os.path.join(os.path.dirname(__file__), '..')
        self.module_name = 'lif_vision'

        # --- LIF vision network configurations ---
        self.max_rate = 63.04
        self.intercept = 0.0
        self.amp = 1.0 / self.max_rate
        self.pstc = 0.005

        # --- LIF vision network weights configurations ---
        # self.vision_network_filename = \
        #     os.path.join(self.filepath, self.module_name, 'params.npz')
        # self.vision_network_data = np.load(self.vision_network_filename,
        #                                    encoding='latin1')

        self.vision_network_data = {}
        self.vision_network_data["weights"] = [np.random.randn(784, 500),
                                               np.random.randn(500, 200)]
        self.vision_network_data["biases"] = [np.random.randn(500),
                                              np.random.randn(200)]
        self.vision_network_data["Wc"] = np.random.randn(200, 24)
        self.vision_network_data["bc"] = np.random.randn(24)

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
        means_filename = \
            os.path.join(self.filepath, self.module_name, 'class_means.npz')
        means_data = np.matrix(1.0 / np.load(means_filename,
                                             encoding='latin1')['means'])

        self.num_classes = weights_class.shape[1]

        self.sps_output_scale = self.amp

        # --- Mandatory data object attributes ---
        self.am_threshold = 0.5

        self.sps = np.array(np.multiply(weights_class.T, means_data.T))
        self.dimensions = self.vision_network_data['Wc'].shape[0]

        self.sps_element_scale = 4.5
        # For magic number 4.5, see reference_code/vision_2/data_analysis.py
