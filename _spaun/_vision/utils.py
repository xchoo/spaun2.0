"""
Denoising autoencoders, single-layer and deep.
"""
import numpy as np
import lif_vision as vision_net


def rms(x, **kwargs):
    return np.sqrt((x ** 2).mean(**kwargs))


def mnist(filepath=''):
    import os
    import urllib

    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, filename=os.path.join(filepath, filename))

    return load_image_data(filename, filepath)


def load_image_data(filename, filepath=''):
    import os
    import gzip
    import cPickle as pickle

    with gzip.open(os.path.join(filepath, filename), 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


def normalize(images):
    """Normalize a set of images"""
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)


class FileObject(object):
    """
    A object that can be saved to file
    """
    def to_file(self, file_name):
        d = {}
        d['__class__'] = self.__class__
        d['__dict__'] = self.__getstate__()
        np.savez(file_name, **d)

    @staticmethod
    def from_file(file_name):
        npzfile = np.load(file_name)
        cls = npzfile['__class__'].item()
        d = npzfile['__dict__'].item()

        self = cls.__new__(cls)
        self.__setstate__(d)
        return self

## TODO: Remove debug mapping once lif vision network has been trained
debug_mapping = {'A': 'SIX', 'OPEN': 'SEV', 'CLOSE': 'EIG', 'QM': 'NIN'}
def get_image(label=None):
    if isinstance(label, int):
        return (vision_net.images_data[label], label)
    elif label is None:
        return (np.zeros(vision_net.images_data_dimensions), -1)
    else:
        #### TODO: remove debug mapping
        if label in debug_mapping.keys():
            label = debug_mapping[label]
        #### END TODO

        label_ind = np.where(vision_net.images_labels_unique == label)
        if label_ind[0].shape[0] > 0:
            image_ind = np.random.choice(
                vision_net.images_labels_inds[label_ind[0][0]])
        else:
            image_ind = np.random.choice(len(vision_net.images_labels_inds))
        return (vision_net.images_data[image_ind], image_ind)
