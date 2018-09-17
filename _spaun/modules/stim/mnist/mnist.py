import gzip
import os
import sys
import numpy as np

# Python2 vs Python3 imports
if sys.version_info[0] < 3:
    import cPickle as pickle
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve
    import pickle

urls = {
    'mnist.pkl.gz': 'http://deeplearning.net/data/mnist/mnist.pkl.gz',
    'spaun_sym.pkl.gz': 'http://files.figshare.com/2106874/spaun_sym.pkl.gz',
}


def read_file(filename, filepath):
    filepath = os.path.join(filepath, filename)
    if not os.path.exists(filepath):
        if filename in urls:
            urlretrieve(urls[filename], filename=filepath)
            print("Fetched '%s' to '%s'" % (urls[filename], filepath))
        else:
            raise NotImplementedError(
                "I do not know where to find '%s'" % filename)

    if sys.version_info[0] < 3:
        with gzip.open(filepath, 'rb') as f:
            train, valid, test = pickle.load(f)
    else:
        with gzip.open(filepath, 'r') as f:
            train, valid, test = pickle.load(f, encoding='bytes')

    return train, valid, test


def load(normalize=False, shuffle=False, spaun=False, seed=8):
    sets = read_file('mnist.pkl.gz')

    if spaun:
        spaun_sets, _, _ = read_file('spaun_sym.pkl.gz')
        spaun_sets = spaun_sets[0][10:], spaun_sets[1][10:]
        sets = _augment(sets[0], sets[1], sets[2],
                        spaun_sets, spaun_sets, spaun_sets)

    if shuffle or spaun:  # always shuffle on augment
        rng = np.random.RandomState(seed)
        sets = tuple(_shuffle(*s, rng=rng) for s in sets)

    if normalize:
        for images, labels in sets:
            _normalize(images)

    return sets


def aug(data, aug_data, ratio):
    images, labels = data
    x, y = aug_data

    n = images.shape[0] / 10  # approximate examples per label
    na = int(n * ratio)       # examples per augmented category

    xx = np.vstack([images, np.tile(x, (na, 1))])
    yy = np.hstack([labels, np.tile(y, na)])

    return xx, yy


def _augment(train, valid, test, aug_train, aug_valid, aug_test, ratio=0.2):
    return aug(train, ratio), aug(valid, ratio), aug(test, ratio)


def _shuffle(images, labels, rng=np.random):
    assert images.shape[0] == labels.shape[0]
    i = rng.permutation(images.shape[0])
    return images[i], labels[i]


def _normalize(images):
    """Normalize a set of images in-place"""
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)
