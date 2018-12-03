import os
import numpy as np
import bisect as bs

from . import mnist


class MNISTDataObject(object):
    def __init__(self, data_filepath=None):
        self.module_name = 'mnist'

        if data_filepath is None:
            self.filepath = os.path.join(os.path.dirname(__file__), '..',
                                         self.module_name)
        else:
            self.filepath = data_filepath

        # --- Mnist data ---
        # _, _, [images_data, images_labels] = \
        #     mnist.read_file('mnist.pkl.gz', self.filepath)
        images_data = np.zeros((10, 28*28))
        images_labels = np.arange(10)
        images_labels = list(map(str, images_labels))

        # --- Spaun symbol data ---
        _, _, [symbol_data, _] = \
            mnist.read_file('spaun_sym.pkl.gz', self.filepath)

        symbol_labels = ['ZER', 'ONE', 'TWO', 'THR', 'FOR', 'FIV', 'SIX',
                         'SEV', 'EIG', 'NIN', 'OPEN', 'CLOSE', 'SPACE', 'QM',
                         'A', 'C', 'F', 'K', 'L', 'M', 'P', 'R', 'V', 'W']

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
        self.images_labels = images_labels
        self.stim_SP_labels = symbol_labels
        self.stim_SP_labels_full = (list(map(str, self.images_labels_unique)) +
                                    symbol_labels)

        self.image_shape = (28, 28)
        self.max_pixel_value = 1.0

        self.probe_subsample = 1
        self.probe_image_shape = (self.image_shape[0] // self.probe_subsample,
                                  self.image_shape[1] // self.probe_subsample)

        subsample_trfm = np.arange(np.cumprod(self.image_shape)[-1])
        subsample_inds = \
            np.array(subsample_trfm.reshape(self.image_shape)
                     [::self.probe_subsample,
                      ::self.probe_subsample].flatten())
        self.probe_subsample_inds = subsample_inds

        self.probe_image_dimensions = subsample_inds.flatten().shape[0]
        self.probe_reset_imgs = \
            [(self.get_image('A')[0] /
             (1.0 * self.max_pixel_value))[subsample_inds],
             (self.get_image('M')[0] /
             (1.0 * self.max_pixel_value))[subsample_inds],
             (self.get_image('V')[0] /
             (1.0 * self.max_pixel_value))[subsample_inds]]

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
                return self.stim_SP_labels_full[label]
        return -1

    def get_image_ind(self, label, rng):
        label_ind = np.where(self.images_labels_unique == label)
        if label_ind[0].shape[0] > 0:
            image_ind = rng.choice(
                self.images_labels_inds[label_ind[0][0]])
        else:
            raise RuntimeError('MNIST - Unable to find label matching ' +
                               '[%s] in label set.' % label)
            # image_ind = rng.choice(len(self.images_labels_inds))
        return image_ind
