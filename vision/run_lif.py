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
def forward(x, weights, biases):
    lif = nengo.LIF()
    for w, b in zip(weights, biases):
        x = np.dot(x, w)
        x += b
        if w is not weights[-1]:
            x = lif.rates(x, 1, 1) / 63.04
    return x

def get_image(t):
    return test_images[int(t / presentation_time)]

def test_dots(t, dots):
    i = int(t / presentation_time)
    j = np.argmax(dots)
    return test_labels[i] == labels[j]

# --- load the RBM data
# data = np.load('nlif-deep-orig.npz')
#data = np.load('nlif-deep.npz')
data = np.load('lif-126-error.npz')
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

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
max_rate = 63.04
intercept = 0
amp = 1. / max_rate
assert np.allclose(neuron_type.gain_bias(max_rate, intercept), (1, 1), atol=1e-2)

model = nengo.Network(seed=97)
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # --- make nonlinear layers
    layers = []
    for i, [W, b] in enumerate(zip(weights[:-1], biases[:-1])):
        n = b.size
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

    # --- make code layer
    W, b = weights[-1], biases[-1]
    code_layer = nengo.networks.EnsembleArray(Ncode, b.size, label='code', radius=5)
    code_bias = nengo.Node(output=b)
    nengo.Connection(code_bias, code_layer.input, synapse=0)
    nengo.Connection(layers[-1].neurons, code_layer.input,
                     transform=W.T * amp * 1000, synapse=pstc)

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(Nclass, 10, label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(code_layer.output, class_layer.input,
                     transform=Wc.T, synapse=pstc)

    test = nengo.Node(output=test_dots, size_in=n_labels)
    nengo.Connection(class_layer.output, test)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    # probe_code = nengo.Probe(code_layer.output, synapse=0.03)
    probe_code = nengo.Probe(code_layer.neuron_output)
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)


# --- simulation
# rundata_file = 'run_lif.npz'
# if not os.path.exists(rundata_file):
if 1:
    sim = nengo.Simulator(model)
    # sim.run(100.)
    sim.run(1.)

    t = sim.trange()
    x = sim.data[probe_code]
    y = sim.data[probe_class]
    z = sim.data[probe_test]

    # np.savez(rundata_file, t=t, y=y, z=z)
else:
    rundata = np.load(rundata_file)
    t, y, z = [rundata[k] for k in ['t', 'y', 'z']]

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

z2 = np.argmax(y, axis=1) == labels.repeat(100)

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
# plt.plot(t, x)
rasterplot(t, sim.data[probe_code][:,:200])
plot_bars()
plt.xticks([])
plt.ylabel('code (500)')

plt.subplot(r, c, 5)
plt.plot(t, y)
plot_bars()
plt.xlabel('time [s]')
plt.ylabel('class')

plt.tight_layout()

# plt.subplot(r, c, 4)
# plt.plot(t, z)
# plt.ylim([-0.1, 1.1])
# plot_bars()
# plt.xlabel('time [s]')
# plt.ylabel('correct')

plt.savefig('run_lif.png')

# --- compute error rate
zblocks = z.reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()

zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()
