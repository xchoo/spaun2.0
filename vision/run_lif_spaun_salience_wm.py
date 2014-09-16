import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

import nengo
from nengo.utils.distributions import Uniform
# from nengo.spa.assoc_mem import AssociativeMemory as AM
from assoc_mem_2_0 import AssociativeMemory as AM
from mem_block import MemoryBlock as MB

# --- parameters
dt = 0.001
present_time = 0.15
# present_time = 0.3
num_digits = 10
# Ncode = 10
# Nclass = 30
Nclass = 50
Nens = 30
# pstc = 0.006
pstc = 0.005

# -- Presentation settings
present_blank = True

max_rate = 63.04
intercept = 0
amp = 1. / max_rate

image_ind = 0

# --- load the RBM data
data = np.load('lif-126-error.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']

# --- load the testing data
from autoencoder import mnist
_, _, [test_images, test_labels] = mnist()
labels = np.unique(test_labels)
n_labels = labels.size

_, _, [sym_images, sym_labels] = mnist('spaun_sym.pkl.gz')

num_classes = Wc.shape[1]

# --- Normalize images
images_mean = test_images.mean(axis=0, keepdims=True)
images_std = 1.0 / np.maximum(test_images.std(axis=0, keepdims=True), 3e-1)

trans_std = np.eye(images_std.shape[1]) * images_std
trans_mean = -np.eye(images_mean.shape[1]) * np.multiply(images_mean,
                                                         images_std)

# --- Shuffle images
import time
seed = int(time.time())
print 'seed: ', seed
seed = 1410806302
rng = np.random.RandomState(seed)
inds = rng.permutation(len(test_images))
inds = inds[range(num_digits)]

test_images = test_images[inds]
test_labels = test_labels[inds]

##
# test_images = np.append(sym_images[10:], test_images, axis=0)

# --- Load mean data
# means = Wc.T * amp / dt / 2
means = Wc.T * amp / dt / 3

# sym_sps = np.load('means_200D.npz')['means']

# means = np.append(means, sym_sps[12:13], axis=0) * 5
# bc = np.append(bc, [0] * sym_sps[12:13].shape[0])
# num_classes = means.shape[0]
# print num_classes


def get_image(t):
    substeps = 10
    if present_blank:
        tmp = t / (present_time / substeps)
        # print t, tmp
        if int(tmp) % substeps >= 5:
        # tmp = t / present_time
        # if int(tmp) != round(tmp):
            return [0] * len(test_images[image_ind])
        else:
            return test_images[image_ind]
    else:
        return test_images[image_ind]


def test_dots(t, dots):
    i = int(t / present_time)
    j = np.argmax(dots)
    return test_labels[i] == labels[j]


def csv_read(filename):
    fobj = open(filename)
    data = []
    for line in fobj.readlines():
        line_info = map(float, line.strip().split(','))
        data.append(line_info)
    return data

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
assert np.allclose(neuron_type.gain_bias(max_rate, intercept), (1, 1),
                   atol=1e-2)

#model = nengo.Network(seed=97)
model = nengo.Network()
with model:
    input_images = nengo.Node(output=get_image, label='images')
    input_bias = nengo.Node(output=[1] * images_mean.shape[1])

    # --- make nonlinear layers
    layers = []
    for i, [W, b] in enumerate(zip(weights, biases)):
        n = b.size
        print "layer %i, size %i" % (i, n)
        layer = nengo.Ensemble(n, 1, label='layer %d' % i,
                               neuron_type=neuron_type,
                               max_rates=max_rate * np.ones(n),
                               intercepts=intercept * np.ones(n))
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.neurons, transform=np.eye(n), synapse=0)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=images_std * W.T, synapse=pstc)
            # nengo.Connection(input_images, layer.neurons,
            #                  transform=images_std * W.T, synapse=0.03)
            nengo.Connection(input_bias, layer.neurons,
                             transform=-np.multiply(images_mean,
                                                    images_std) * W.T,
                             synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp * 1000, synapse=pstc)

        layers.append(layer)

    threshold = 0.8

    am = AM(means, [[1] for i in range(num_classes)],
            output_utilities=True, output_thresholded_utilities=True,
            wta_output=True, wta_inhibit_scale=3.0, threshold=threshold,
            default_output_vector=[1])
    # nengo.Connection(layers[-1].neurons, am.input, synapse=0.02)
    nengo.Connection(layers[-1].neurons, am.input, synapse=0.005)

    am2 = AM(means, [[1] for i in range(num_classes)],
             output_utilities=True, output_thresholded_utilities=True,
             neuron_type=nengo.Direct(), threshold=threshold)
    # nengo.Connection(layers[-1].neurons, am2.input, synapse=0.02)
    nengo.Connection(layers[-1].neurons, am2.input, synapse=0.005)

    # --- add biases to cleanup?
    bias = nengo.Node(output=1)
    am.add_input('bias', [[1]] * num_classes, input_scale=bc)
    am2.add_input('bias', [[1]] * num_classes, input_scale=bc)
    nengo.Connection(bias, am.bias)
    nengo.Connection(bias, am2.bias)

    # Salience bits
    sal = nengo.networks.EnsembleArray(Nens, layers[-1].n_neurons,
                                       label='salience',
                                       intercepts=Uniform(0.1, 1))
    # sal = nengo.networks.EnsembleArray(1, layers[-1].n_neurons,
    #                                    label='salience',
    #                                    neuron_type=nengo.Direct())
    sal.add_output('abs', lambda x: abs(x))
    nengo.Connection(layers[-1].neurons, sal.input, synapse=0.005,
                     transform=3.5)
    nengo.Connection(layers[-1].neurons, sal.input, synapse=0.03,
                     transform=-3.5)

    sal_node = nengo.Node(size_in=1)
    nengo.Connection(sal.abs, sal_node,
                     transform=[[1] * layers[-1].n_neurons])
    sal_trig = AM([[1]], threshold=0.3, input_scale=2.0)
    nengo.Connection(sal.abs, sal_trig.input,
                     transform=[[1] * layers[-1].n_neurons])

    ## Inhibit salience on space?
    neg_sal = nengo.Ensemble(30, 1, intercepts=Uniform(0.5, 1))
    # neg_sal introduces a delay
    nengo.Connection(am.default_output_utility, neg_sal, synapse=0.03)
    for ens in sal.ensembles:
        nengo.Connection(neg_sal, ens.neurons,
                         transform=[[-5]] * ens.n_neurons, synapse=0.005)

    ## Memory Block bits
    mb_delay = nengo.networks.EnsembleArray(50, num_classes)
    nengo.Connection(am.thresholded_utilities, mb_delay.input, synapse=0.02)

    mb1 = MB(dimensions=num_classes, radius=1, mem_synapse=0.05)
    nengo.Connection(mb_delay.output, mb1.input, synapse=0.02)
    nengo.Connection(sal_trig.output, mb1.gate)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    probe_am_tu = nengo.Probe(am.thresholded_utilities, synapse=0.005)
    # probe_am_def = nengo.Probe(am.default_output_utility, synapse=0.005)
    probe_am_def = nengo.Probe(neg_sal, synapse=0.005)
    probe_am2_u = nengo.Probe(am2.utilities, synapse=0.005)
    probe_am2_tu = nengo.Probe(am2.thresholded_utilities, synapse=0.03)
    probe_sal = nengo.Probe(sal_node, synapse=0.03)
    probe_salt = nengo.Probe(sal_trig.output, synapse=0.005)
    probe_sal2 = nengo.Probe(sal.output, synapse=0.03)
    probe_mb1 = nengo.Probe(mb1.mem1.output, synapse=0.005)
    probe_mb2 = nengo.Probe(mb1.mem2.output, synapse=0.005)
    probe_mb3 = nengo.Probe(mb1.mem1.diff.output, synapse=0.005)
    probe_mb4 = nengo.Probe(mb1.mem2.diff.output, synapse=0.005)

sim = nengo.Simulator(model)

# --- simulation
for n in range(num_digits):
    print "Processing digit %i of %i" % (n + 1, num_digits)
    image_ind = n
    sim.run(present_time)

# --- Plots
# from nengo.utils.matplotlib import rasterplot


def plot_bars():
    ylim = plt.ylim()
    for x in np.arange(0, t[-1], present_time):
        plt.plot([x, x], ylim, 'k--')

t = sim.trange()

inds = slice(0, int(t[-1] / present_time) + 1)
images = test_images[inds]
# labels = test_labels[inds]
allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
for i, image in enumerate(images):
    allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

# plt.figure(1)
plt.clf()
r, c = 3, 1
# r, c = 7, 1
# r, c = 6, 1

p = 1
plt.subplot(r, c, p)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
plt.xticks([])
plt.yticks([])

# p += 1
# plt.subplot(r, c, p)
# rasterplot(t, sim.data[probe_layers[0]][:, :200])
# plot_bars()
# plt.xticks([])
# plt.xlim([t[0], t[-1]])
# plt.ylabel('layer 1 (500)')

# p += 1
# plt.subplot(r, c, p)
# rasterplot(t, sim.data[probe_layers[1]])
# plt.xticks([])
# plt.yticks(np.linspace(0, 200, 5))
# plot_bars()
# plt.xlim([t[0], t[-1]])
# plt.ylabel('layer 2 (200)')

p += 1
colormap = plt.cm.gist_ncar
plt.subplot(r, c, p)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
for i in range(num_classes):
    plt.plot(t, sim.data[probe_am_tu][:, i])
plt.xlim([t[0], t[-1]])
plt.ylabel('am u')

# p += 1
# plt.subplot(r, c, p)
# plt.plot(t, sim.data[probe_am_def])
# plt.xlim([t[0], t[-1]])

p += 1
plt.subplot(r, c, p)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
for i in range(num_classes):
    plt.plot(t, sim.data[probe_am2_u][:, i])
plt.xlim([t[0], t[-1]])
plt.ylabel('am ttu')

# p += 1
# plt.subplot(r, c, p)
# plt.plot(t, sim.data[probe_am2_u][:, 10])
# plt.xlim([t[0], t[-1]])

# p += 1
# plt.subplot(r, c, p)
# plt.plot(t, sim.data[probe_sal])
# plt.plot(t, sim.data[probe_salt])
# # plt.plot(t, sim.data[probe_sal2])
# plt.xlim([t[0], t[-1]])

# p += 1
# plt.subplot(r, c, p)
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
# for i in range(num_classes):
#     plt.plot(t, sim.data[probe_mb1][:, i])
# plt.ylabel('mb1')
# plt.xlim([t[0], t[-1]])

# p += 1
# plt.subplot(r, c, p)
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
# for i in range(num_classes):
#     plt.plot(t, sim.data[probe_mb2][:, i])
# plt.ylabel('mb2')
# plt.xlim([t[0], t[-1]])

# p += 1
# plt.subplot(r, c, p)
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
# for i in range(num_classes):
#     plt.plot(t, sim.data[probe_mb3][:, i])
# plt.ylabel('mb3')
# plt.xlim([t[0], t[-1]])

# p += 1
# plt.subplot(r, c, p)
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
# for i in range(num_classes):
#     plt.plot(t, sim.data[probe_mb4][:, i])
# plt.ylabel('mb4')
# plt.xlim([t[0], t[-1]])

plt.tight_layout()
# plt.show()
plt.savefig('test_am.png')

import os
os.system('test_am.png')
