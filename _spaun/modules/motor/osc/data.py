import os
import numpy as np

from ..data import MotorDataObject


class OSCMotorDataObject(MotorDataObject):
    def __init__(self):
        super(OSCMotorDataObject, self).__init__()
        self.module_name = 'osc'

        # --- Controller network configuration ---
        self.kp = 65
        self.kv1 = np.sqrt(8)
        self.kv2 = np.sqrt(18) - self.kv1
        # KV2 - Additional KV to use when moving arm to start of trajectory

        # --- Motor semantic pointer generation ---
        canonical_paths = np.load(os.path.join(self.filepath,
                                               self.module_name,
                                               'canon_paths.npz'),
                                  encoding='latin1')
        canonical_paths_x = canonical_paths['canon_paths_x']
        canonical_paths_y = canonical_paths['canon_paths_y']

        self.dimensions = (canonical_paths_x.shape[1] +
                           canonical_paths_y.shape[1])

        self.num_sps = canonical_paths_x.shape[0]

        # Function that converts path (x,y loc) information to difference of
        # points information
        def make_mtr_sp(path_x, path_y):
            return np.concatenate((path_x, path_y))

        self.sps = np.zeros((self.num_sps, self.dimensions))
        for n in range(self.num_sps):
            self.sps[n, :] = make_mtr_sp(canonical_paths_x[n, :],
                                         canonical_paths_y[n, :])

        self.sp_scaling_factor = \
            float(canonical_paths['size_scaling_factor'])
