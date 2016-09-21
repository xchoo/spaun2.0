"""
Denoising autoencoders, single-layer and deep.
"""
import numpy as np


def rms(x, **kwargs):
    return np.sqrt((x ** 2).mean(**kwargs))


def mnist(filepath=''):
    import os
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    filename = 'mnist.pkl.gz'
    if not os.path.exists(os.path.join(filepath, filename)):
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        urlretrieve(url, filename=os.path.join(filepath, filename))

    return load_image_data(filename, filepath)


def load_image_data(filename, filepath=''):
    import os
    import gzip
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    with gzip.open(os.path.join(filepath, filename), 'r') as f:
        train, valid, test = pickle.load(f, encoding='bytes')

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
