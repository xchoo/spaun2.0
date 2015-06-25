"""
Reads the spaun vision data stored in pkl files and writes
it to a csv file, which can be more easily read from C++
"""

import numpy as np
import os
import urllib
import csv

home = os.getenv("HOME")

data_path = home + '/spaun2.0/_spaun/vision/'
dest_path = data_path + 'spaun_vision_data.csv'


def load_image_data(filename):
    import gzip
    import cPickle as pickle

    with gzip.open(filename, 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


mnist_filename = os.path.join(data_path, 'mnist.pkl.gz')

if not os.path.exists(mnist_filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=mnist_filename)

_, _, [data, labels] = load_image_data(mnist_filename)

labels = map(str, labels)

symbols_filename = os.path.join(data_path, 'spaun_sym.pkl.gz')
_, _, [symbol_data, symbol_labels] = load_image_data(symbols_filename)

data = np.append(data, symbol_data, axis=0)
labels = np.append(labels, symbol_labels, axis=0)

with open(dest_path, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')

    for l, d in zip(labels, data):
        writer.writerow(l)
        writer.writerow(d)
