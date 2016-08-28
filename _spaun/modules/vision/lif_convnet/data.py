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

        self.okey = 'fc1'

        # --- Visual associative memory configurations ---
        means_filename = \
            os.path.join(self.filepath, self.module_name, 'class_centers.npz')

        sps = np.load(means_filename)['centers']
        self.num_classes = sps.shape[0]

        self.vis_net_output_scale = 1.0 / 32.0
        # For magic number 32.0, see reference_code/vision_3/data_analysis.py
        # (print np.mean(norms))

        # --- Mandatory data object attributes ---
        self.am_threshold = 0.5
        self.sps = sps
        self.dimensions = self.sps.shape[1]

        self.sps_element_scale = 1.0 * self.vis_net_output_scale
        # For magic number 1.0, see reference_code/vision_3/data_analysis.py
        # (95% of sp_2000 data covered by sp_2000 between [0, 1.0))
