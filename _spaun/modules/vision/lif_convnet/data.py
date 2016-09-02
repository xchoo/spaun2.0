import os
import numpy as np

from nengo_extras.cuda_convnet import load_model_pickle
from ..data import VisionDataObject


class LIFConvNetVisionDataObject(VisionDataObject):
    def __init__(self):
        super(LIFConvNetVisionDataObject, self).__init__()
        self.module_name = 'lif_convnet'

        # --- LIF vision network weights configurations ---
        self.vision_network_filename = \
            os.path.join(self.filepath, self.module_name, 'params.pkl')
        self.vision_network_data = \
            load_model_pickle(self.vision_network_filename)

        self.mem_okey = 'fc1'
        self.classify_okey = 'fc10'

        # --- Visual associative memory configurations ---
        self.am_threshold = 0.5

        centers_filename = \
            os.path.join(self.filepath, self.module_name, 'class_centers.npz')

        sps = np.load(centers_filename)['centers']
        self.num_classes = sps.shape[0]

        self.sps_output_scale = 1.0 / 32.0
        # For magic number 32.0, see reference_code/vision_3/data_analysis.py
        # (print np.mean(norms) for sp_2000)

        means_filename = \
            os.path.join(self.filepath, self.module_name,
                         'fc10_class_means.npz')

        self.sps_fc10 = np.eye(self.num_classes)
        self.sps_fc10_means = np.load(means_filename)['means']
        # For magic number 10.1, see reference_code/vision_3/data_analysis.py
        # (print np.mean(norms) for sp_10)

        # --- Mandatory data object attributes ---
        self.sps = sps
        self.dimensions = self.sps.shape[1]

        self.sps_element_scale = 1.0 * self.sps_output_scale
        # For magic number 1.0, see reference_code/vision_3/data_analysis.py
        # (95% of sp_2000 data covered by sp_2000 between [0, 1.0))
