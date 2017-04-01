import os
import numpy as np
import bisect as bs

from nengo_extras.data import load_ilsvrc2012, spasafe_names


class ImagenetDataObject(object):
    def __init__(self):
        self.module_name = 'imagenet'
        self.filepath = os.path.join('_spaun', 'modules', 'stim',
                                     self.module_name)

        # --- Spaun symbol data ---
        spaun_sym_filename = 'spaun_sym.npz'

        sym_fileobj = np.load(os.path.join(self.filepath, spaun_sym_filename))

        # --- Imagenet data ---
        data_filename = 'ilsvrc-2012-batches-test3.tar.gz'

        images_data, images_labels, images_data_mean, class_labels = \
            load_ilsvrc2012(os.path.join(self.filepath, data_filename),
                            n_files=1)

        # --- Mean data ---
        data_mean_filename = 'image_data_mean.npz'

        # Image pre-processing
        images_data = images_data.astype('float32')
        images_data = images_data[:, :, 16:-16, 16:-16]
        # images_data_mean = images_data_mean[:, 16:-16, 16:-16]
        images_data_mean = \
            np.load(os.path.join(self.filepath,
                                 data_mean_filename))['data_mean']

        # --- Combined image (imagenet + spaun symbol) data ---
        # Spaun symbol data
        sym_im_data = sym_fileobj['image_data']
        sym_label_strs = list(sym_fileobj['image_labels'])

        self.num_spaun_sym_classes = len(sym_label_strs)
        sym_im_labels = range(self.num_spaun_sym_classes)

        # Imagenet data
        self.images_data_mean = images_data_mean.flatten()

        self.images_data_dimensions = \
            images_data[0, :, :, :].flatten().shape[0]

        sorted_inds = np.argsort(images_labels)
        imagenet_im_data = images_data[sorted_inds]
        imagenet_im_labels = np.array(images_labels[sorted_inds])

        imagenet_label_strs = list(map(lambda s: str(s).upper(),
                                   spasafe_names(class_labels)))

        self.num_imagenet_classes = len(imagenet_label_strs)

        # Combined data
        self.images_data = np.vstack((sym_im_data, imagenet_im_data))
        self.images_labels = (sym_im_labels +
                              list(imagenet_im_labels +
                                   self.num_spaun_sym_classes))
        self.stim_SP_labels = np.array(sym_label_strs + imagenet_label_strs)
        self.num_classes = (self.num_spaun_sym_classes +
                            self.num_imagenet_classes)

        # Separate image data into a list for each image label (class)
        self.images_labels_inds = [None] * self.num_classes
        self.images_labels_unique = np.unique(self.images_labels)
        for lbl in self.images_labels_unique:
            self.images_labels_inds[lbl] = \
                range(bs.bisect_left(self.images_labels, lbl),
                      bs.bisect_right(self.images_labels, lbl))

        # --- Image data ---
        self.image_shape = (3, 224, 224)
        self.max_pixel_value = 255.0

        # --- Hack in Spaun labels ---
        # from collections import OrderedDict
        # self.symbol_map = \
        #     OrderedDict([('ZER', 0), ('ONE', 2), ('TWO', 4), ('THR', 8),
        #                  ('FOR', 9), ('FIV', 14), ('SIX', 19), ('SEV', 21),
        #                  ('EIG', 22), ('NIN', 23), ('CLOSE', 25), ('OPEN', 26),
        #                  ('SPACE', 32), ('QM', 35), ('W', 37), ('V', 49),
        #                  ('R', 50), ('P', 52), ('M', 54), ('L', 55),
        #                  ('K', 57), ('F', 59), ('C', 60), ('A', 65)])
        # for SP_str in self.symbol_map.keys():
        #     image_ind = int(images_labels[self.symbol_map[SP_str]])
        #     self.stim_SP_labels[image_ind] = SP_str

        # --- Handle subsampling of probe data ---
        self.probe_subsample = 4
        self.probe_image_shape = (self.image_shape[0],
                                  self.image_shape[1] / self.probe_subsample,
                                  self.image_shape[2] / self.probe_subsample)

        subsample_trfm = np.arange(np.cumprod(self.image_shape)[-1])
        subsample_inds = \
            np.array(subsample_trfm.reshape(self.image_shape)
                     [:, ::self.probe_subsample,
                      ::self.probe_subsample].flatten())
        self.probe_subsample_inds = subsample_inds

        self.probe_image_dimensions = subsample_inds.flatten().shape[0]
        self.probe_reset_img = (self.get_image('A')[0] /
                                (1.0 * self.max_pixel_value))[subsample_inds]

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
            return (self.images_data[image_ind].flatten(), image_ind)

    def get_image_label(self, index):
        for label, indicies in enumerate(self.images_labels_inds):
            if indicies is not None and index in indicies:
                return self.stim_SP_labels[label]
        return -1

    def get_image_ind(self, label, rng):
        # --- HACK FOR SPAUN DIGITS
        # if label in self.symbol_map.keys():
        #     return self.symbol_map[label]
        # ---

        if all(c.isdigit() for c in label):
            # Case where the class label index (and not the SP name) is given
            label_ind = (np.array([int(label)]),)
        else:
            # Case where class label SP name is given
            label_ind = np.where(self.stim_SP_labels == label)

        if label_ind[0].shape[0] > 0:
            if label_ind[0][0] is None:
                image_ind = 0   # TODO: Fix this? This happens when trying to
                # get image that was not in dataset (depends on nfiles)
            else:
                image_ind = rng.choice(
                    self.images_labels_inds[label_ind[0][0]])
        else:
            raise RuntimeError('IMAGENET - Unable to find label matching ' +
                               '[%s] in label set.' % label)
            # image_ind = rng.choice(len(self.images_labels_inds))
        return image_ind
