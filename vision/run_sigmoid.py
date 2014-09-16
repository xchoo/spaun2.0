import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import nengo

# --- parameters
presentation_time = 0.1
Ncode = 10
Nclass = 30
# pstc = 0.006
pstc = 0.004

# --- functions
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def forward(x, weights, biases):
    for w, b in zip(weights, biases):
        x = np.dot(x, w)
        x += b
        if w is not weights[-1]:
            x = sigmoid(x)
    return x

def get_image(t):
    return test_images[int(t / presentation_time)]

def test_dots(t, dots):
    i = int(t / presentation_time)
    j = np.argmax(dots)
    return test_labels[i] == labels[j]

# --- load the RBM data
data = np.load('sigmoid-deep.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']

# --- load the testing data
from autoencoder import mnist
_, _, [test_images, test_labels] = mnist()

for images in [test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# shuffle
rng = np.random.RandomState(92)
inds = rng.permutation(len(test_images))
test_images = test_images[inds]
test_labels = test_labels[inds]

labels = np.unique(test_labels)
n_labels = labels.size

# --- test as ANN
codes = forward(test_images, weights, biases)
inds = np.argmax(np.dot(codes, Wc) + bc, axis=1)
labels = np.unique(test_labels)
errors = (test_labels != labels[inds])
print "ANN error:", errors.mean()

# --- find good neuron params for sigmoid
import find_neuron_params

neuron_params_file = 'neuron_params.npz'
if not os.path.exists(neuron_params_file):
    find_neuron_params.find_params(savefile=neuron_params_file, show=False)

neuron_params = dict(np.load(neuron_params_file))
N = neuron_params.pop('N')

# --- create the model
model = nengo.Network()
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # --- make sigmoidal layers
    layers = []
    output = input_images
    for w, b in zip(weights[:-1], biases[:-1]):
        layer = nengo.networks.EnsembleArray(N, b.size, **neuron_params)
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.input, synapse=0)

        nengo.Connection(output, layer.input, transform=w.T, synapse=pstc)
        output = layer.add_output('sigmoid', function=sigmoid)

        layers.append(layer)

    # --- make code layer
    W, b = weights[-1], biases[-1]
    code_layer = nengo.networks.EnsembleArray(Ncode, b.size, label='code', radius=10)
    code_bias = nengo.Node(output=b)
    nengo.Connection(code_bias, code_layer.input, synapse=0)
    nengo.Connection(output, code_layer.input, transform=W.T, synapse=pstc)

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(Nclass, 10, label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(code_layer.output, class_layer.input,
                     transform=Wc.T, synapse=pstc)

    test = nengo.Node(output=test_dots, size_in=n_labels)
    nengo.Connection(class_layer.output, test)

    probe_code = nengo.Probe(code_layer.output, synapse=0.03)
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)


# --- simulation
# rundata_file = 'rundata.npz'
# if not os.path.exists(rundata_file):
if 1:
    sim = nengo.Simulator(model)
    sim.run(100.)
    # sim.run(10.)

    t = sim.trange()
    x = sim.data[probe_code]
    y = sim.data[probe_class]
    z = sim.data[probe_test]

    # np.savez(rundata_file, t=t, y=y, z=z)
else:
    rundata = np.load(rundata_file)
    t, y, z = [rundata[k] for k in ['t', 'y', 'z']]

# --- plots
def plot_bars():
    ylim = plt.ylim()
    for x in np.arange(0, t[-1], presentation_time):
        plt.plot([x, x], ylim, 'k--')

inds = slice(0, int(t[-1]/presentation_time) + 1)
images = test_images[inds]
labels = test_labels[inds]
allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
for i, image in enumerate(images):
    allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

z2 = np.argmax(y, axis=1) == labels.repeat(100)

plt.figure(1)
plt.clf()
r, c = 4, 1

plt.subplot(r, c, 1)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(r, c, 2)
plt.plot(t, x)
plot_bars()
plt.ylabel('code')

plt.subplot(r, c, 3)
plt.plot(t, y)
plot_bars()
plt.ylabel('class')

plt.subplot(r, c, 4)
plt.plot(t, z)
plt.ylim([-0.1, 1.1])
plot_bars()
plt.xlabel('time [s]')
plt.ylabel('correct')

# plt.savefig('runtime.png')

# --- compute error rate
zblocks = z.reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()

zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()
