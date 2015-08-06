"""
Reads the spaun vision data stored in pkl files and writes
it to a csv file, which can be more easily read from C++
"""

import numpy as np
import os
import urllib
import shutil
import csv
import collections

home = os.getenv("HOME")

vision_dir = home + '/spaun2.0/_spaun/vision/'


def load_image_data(filename):
    import gzip
    import cPickle as pickle

    with gzip.open(filename, 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


def image_string(image):
    s = int(np.sqrt(len(image)))
    string = ""
    for i in range(s):
        for j in range(s):
            string += str(1 if image[i * s + j] > 0 else 0)
        string += '\n'
    return string


mnist_filename = os.path.join(vision_dir, 'mnist.pkl.gz')

if not os.path.exists(mnist_filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=mnist_filename)

_, _, [data, labels] = load_image_data(mnist_filename)

labels = map(str, labels)

symbols_filename = os.path.join(vision_dir, 'spaun_sym.pkl.gz')
_, _, [symbol_data, symbol_labels] = load_image_data(symbols_filename)

data = np.append(data, symbol_data, axis=0)
labels = np.append(labels, symbol_labels, axis=0)

labelled_data = zip(labels, data)

format = 'folder'
if format == 'folder':
    data_path = os.path.join(vision_dir, 'spaun_vision_data')

    if os.path.isdir(data_path):
        shutil.rmtree(data_path)

    os.makedirs(data_path)

    data_dict = collections.OrderedDict()
    lbl_dirs = []
    for lbl in list(set(labels)):
        lbl_dir = os.path.join(data_path, lbl)
        if not os.path.isdir(lbl_dir):
            os.makedirs(lbl_dir)
        lbl_dirs.append(lbl_dir)

        data_dict[lbl] = [d for (l, d) in labelled_data if l == lbl]

    for lbl_dir, (lbl, dlist) in zip(lbl_dirs, data_dict.iteritems()):
        print "Processing imgs with label %s" % lbl

        assert lbl in lbl_dir
        for i, d in enumerate(dlist):
            with open(os.path.join(lbl_dir, str(i)), 'wb') as csvfile:
                writer = csv.writer(
                    csvfile, delimiter=',', lineterminator='\n')
                writer.writerow(d)

    print "Writing counts file"
    with open(os.path.join(data_path, 'counts'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        for lbl, dlist in data_dict.iteritems():
            writer.writerow([lbl])
            writer.writerow([len(dlist)])
else:
    data_path = os.path.join(vision_dir, 'spaun_vision_data.csv')

    with open(data_path, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        for l, d in zip(labels, data):
            writer.writerow(l)
            writer.writerow(d)
