import os
import numpy as np

from nengo_extras.cuda_convnet import load_model_pickle


class LIFImagenetVisionDataObject(object):
    def __init__(self):
        self.filepath = os.path.join(os.path.dirname(__file__), '..')
        self.module_name = 'lif_imagenet'  # Use the lif_imagenet vision files

        # --- LIF vision network weights configurations ---
        self.vision_network_filename = \
            os.path.join(self.filepath, self.module_name, 'params.pkl')

        # Try alternate params filename if can't locate default file
        if not os.path.exists(self.vision_network_filename):
            self.vision_network_filename = \
                os.path.join(self.filepath, self.module_name,
                             'ilsvrc2012-lif-48.pkl')

        self.vision_network_data = \
            load_model_pickle(self.vision_network_filename)

        # --- Spaun symbol associative memory configurations ---
        self.spaun_am_threshold = 0.5
        self.spaun_am_input_bias = -9.0
        self.spaun_am_input_scale = 10.0

        centers_filename = \
            os.path.join(self.filepath, self.module_name,
                         'spaun_sym_class_centers.npz')
        self.sps_spaun_sym = np.load(centers_filename,
                                     encoding='latin1')['class_centers']
        self.spaun_sym_num_classes = self.sps_spaun_sym.shape[0]
        self.spaun_sym_out_dimensions = self.sps_spaun_sym.shape[1]

        # --- Imagenet associative memory configurations ---
        # self.imagenet_am_threshold = 0.5
        # self.imagenet_am_threshold = 0.30
        self.imagenet_am_threshold = 0.21

        means_filename = \
            os.path.join(self.filepath, self.module_name,
                         'class_means.npz')

        self.imagenet_sps_means = np.load(means_filename,
                                          encoding='latin1')['means']
        self.imagenet_num_classes = self.imagenet_sps_means.shape[0]
        self.imagenet_out_dimensions = self.imagenet_num_classes

        self.sps_fc1000 = np.eye(self.imagenet_num_classes)

        # --- Combined configurations ---
        # Here, we are going to output layer 'pool2' to the spaun_sym AM
        # (output dimensions = self.spaun_sym_out_dimensions) + 'fc1000' to
        # imagenet AM (output_dimensions = self.imagenet_out_dimensions)
        self.spaun_sym_okey = 'pool2'
        self.output_dimensions = (self.spaun_sym_out_dimensions +
                                  self.imagenet_out_dimensions)

        # --- Mandatory data object attributes ---
        self.sps = np.zeros((self.spaun_sym_num_classes +     # HACK: To avoid
                             self.imagenet_num_classes, 10))  # creating vis_wm
        self.dimensions = self.sps.shape[1]                   # of 1000+ dims

        self.sps_element_scale = 1.0
        self.am_threshold = 0.5
