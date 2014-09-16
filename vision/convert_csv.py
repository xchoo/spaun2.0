import numpy as np

from autoencoder import mnist
train, valid, test = mnist('spaun_sym.pkl.gz')
print train[0].shape
print train[1].shape
print valid[0].shape
print valid[1].shape


def csv_read(filename, map_func=float):
    fobj = open(filename)
    data = []
    for line in fobj.readlines():
        line_info = map_func(line.strip().split(','))
        data.append(line_info)
    return data

sym_list = csv_read('sym_list.csv', lambda l: l[0])
sym_ind = range(len(sym_list))
sym_vis = csv_read('sym_vis.csv', lambda l: map(float, l))

sym_vis = np.array(sym_vis)
sym_ind = np.array(sym_ind)


print type(sym_vis)
print type(sym_ind)
print sym_vis.shape
print sym_ind.shape

new_data = (sym_vis, sym_ind)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(train[0][0].reshape(28, 28), cmap=cm.Greys_r)
plt.subplot(2, 1, 2)
plt.imshow(sym_vis[13].reshape(28, 28), cmap=cm.Greys_r)
plt.show()


import gzip
import cPickle as pickle

data = (new_data, new_data, new_data)

f = gzip.open('spaun_sym.pkl.gz', 'wb')
pkl_dump = pickle.dumps(data)
f.writelines(pkl_dump)
f.close()
