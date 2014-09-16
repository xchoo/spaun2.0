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
# Nclass = 30
Nclass = 50
# pstc = 0.006
pstc = 0.005

# --- functions
def forward(x, weights, biases):
    lif = nengo.LIF()
    layers = []
    for w, b in zip(weights, biases):
        x = np.dot(x, w) + b
        x = lif.rates(x, 1, 1) / 63.04
        layers.append(x)
    return x, layers

def get_image(t):
    return test_images[int(t / presentation_time)]

def test_dots(t, dots):
    i = int(t / presentation_time)
    j = np.argmax(dots)
    return test_labels[i] == labels[j]

def csv_read(filename):
    fobj = open(filename)
    data = []
    for line in fobj.readlines():
        line_info = map(float, line.strip().split(','))
        data.append(line_info)
    return data


# --- load the RBM data
# data = np.load('nlif-deep-orig.npz')
# data = np.load('lif-500-200-10.npz')
data = np.load('lif-126-error.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']

# --- load the testing data
from autoencoder import mnist
_, _, [test_images, test_labels] = mnist()

# for images in [test_images]:
#     images -= images.mean(axis=0, keepdims=True)
#     images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# shuffle
rng = np.random.RandomState(None)
inds = rng.permutation(len(test_images))
test_images = test_images[inds]
test_labels = test_labels[inds]

labels = np.unique(test_labels)
n_labels = labels.size

# --- test as ANN
codes, layers = forward(test_images, weights, biases)
classes = np.dot(codes, Wc) + bc
inds = np.argmax(classes, axis=1)
labels = np.unique(test_labels)
errors = (test_labels != labels[inds])
print "ANN error:", errors.mean()
print "ANN classes: mean=%0.3f, std=%0.3f, min=%0.3f, max=%0.3f" % (
    classes.mean(), classes.std(0).mean(), classes.min(), classes.max())
for i, layer in enumerate(layers):
    print "Layer %d: sparsity=%0.3f, %0.3f" % (i, (layer > 0).mean(), (layer > 1).mean())

if 1:
    plt.figure(101)
    plt.clf()
    r = len(layers)
    for i, layer in enumerate(layers):
        plt.subplot(r, 1, i+1)
        plt.hist(layer.flatten(), bins=15)

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
max_rate = 63.04
intercept = 0
amp = 1. / max_rate
assert np.allclose(neuron_type.gain_bias(max_rate, intercept), (1, 1), atol=1e-2)

#model = nengo.Network(seed=97)
model = nengo.Network()
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # --- make nonlinear layers
    layers = []
    for i, [W, b] in enumerate(zip(weights, biases)):
        n = b.size
        print "layer %i, size %i" % (i, n)
        layer = nengo.Ensemble(n, 1, label='layer %d' % i, neuron_type=neuron_type,
                               max_rates=max_rate*np.ones(n),
                               intercepts=intercept*np.ones(n))
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.neurons, transform=np.eye(n), synapse=0)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp * 1000, synapse=pstc)

        layers.append(layer)

    print data.keys()
    print W.shape
    print data['rec_weights'].shape
    print Wc.shape
    print bc.shape
    print bc
    print Nclass

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(Nclass, 10, label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(layers[-1].neurons, class_layer.input,
                     transform=Wc.T * amp * 1000, synapse=pstc)

    test = nengo.Node(output=test_dots, size_in=n_labels)
    nengo.Connection(class_layer.output, test)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)


# --- simulation
sim = nengo.Simulator(model)
#sim.run(100.)
sim.run(2.)

t = sim.trange()

# --- plots
from nengo.utils.matplotlib import rasterplot

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

plt.figure(1)
plt.clf()
r, c = 5, 1

plt.subplot(r, c, 1)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(r, c, 2)
rasterplot(t, sim.data[probe_layers[0]][:,:200])
plot_bars()
plt.xticks([])
plt.ylabel('layer 1 (500)')

plt.subplot(r, c, 3)
rasterplot(t, sim.data[probe_layers[1]])
plt.xticks([])
plt.yticks(np.linspace(0, 200, 5))
plot_bars()
plt.ylabel('layer 2 (200)')

plt.subplot(r, c, 4)
plt.plot(t, sim.data[probe_class])
plot_bars()
plt.xlabel('time [s]')
plt.ylabel('class')

plt.subplot(r, c, 5)
plt.plot(t, sim.data[probe_test])
plt.ylim([-0.1, 1.1])
plot_bars()
plt.xlabel('time [s]')
plt.ylabel('correct')

plt.tight_layout()

plt.savefig('run_lif.png')

# --- compute error rate
zblocks = sim.data[probe_test].reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()

# z2 = np.argmax(y, axis=1) == labels.repeat(100)
# zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
# errors = np.mean(zblocks, axis=1) < 0.5
# print errors.mean()
