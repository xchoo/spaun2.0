import os
import numpy as np

from ..._networks import convert_func_2_diff_func


class MotorDataObject(object):
    def __init__(self):
        self.filepath = os.path.join('_spaun', 'modules', 'motor')

        canonical_paths = np.load(os.path.join(self.filepath,
                                               'canon_paths.npz'))
        canonical_paths_x = canonical_paths['canon_paths_x']
        canonical_paths_y = canonical_paths['canon_paths_y']

        self.dimensions = (canonical_paths_x.shape[1] +
                           canonical_paths_y.shape[1])

        self.num_sps = canonical_paths_x.shape[0]

        # Function that converts path (x,y loc) information to difference of
        # points information
        def make_mtr_sp(path_x, path_y):
            # path_x = convert_func_2_diff_func(path_x)
            # path_y = convert_func_2_diff_func(path_y)
            return np.concatenate((path_x, path_y))

        self.sps = np.zeros((self.num_sps, self.dimensions))
        for n in range(self.num_sps):
            self.sps[n, :] = make_mtr_sp(canonical_paths_x[n, :],
                                         canonical_paths_y[n, :])

        self.sp_scaling_factor = \
            float(canonical_paths['size_scaling_factor'])

mtr_data = MotorDataObject()
