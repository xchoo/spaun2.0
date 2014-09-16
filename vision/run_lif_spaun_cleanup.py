import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

import nengo
# from nengo.spa.assoc_mem import AssociativeMemory as AM
from assoc_mem_2_0 import AssociativeMemory as AM

# --- parameters
dt = 0.001
present_time = 0.15
num_digits = 15
# Ncode = 10
# Nclass = 30
Nclass = 50
Nnorm = 50
# pstc = 0.006
pstc = 0.005

max_rate = 63.04
intercept = 0
amp = 1. / max_rate

image_ind = 0

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
# _, _, [test_images, test_labels] = mnist('spaun_sym.pkl.gz')
labels = np.unique(test_labels)
n_labels = labels.size

num_classes = Wc.shape[1]

# --- Normalize images
# for images in [test_images]:
#     print images.shape
#     images -= images.mean(axis=0, keepdims=True)
#     images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)
images_mean = test_images.mean(axis=0, keepdims=True)
images_std = 1.0 / np.maximum(test_images.std(axis=0, keepdims=True), 3e-1)

trans_std = np.eye(images_std.shape[1]) * images_std
trans_mean = -np.eye(images_mean.shape[1]) * np.multiply(images_mean,
                                                         images_std)

# test_images -= images_mean
# test_images /= images_std

# --- Shuffle images
rng = np.random.RandomState(1)
# rng = np.random.RandomState(None)
inds = rng.permutation(len(test_images))
inds = inds[range(num_digits)]
# num_digits = n_labels
# inds = range(num_digits)
test_images = test_images[inds]
test_labels = test_labels[inds]

# --- Load mean data
# means = np.load('means_200D.npz')['means']
means = Wc.T * amp / dt / 4
scales = np.load('scales_200D.npz')['scales']
# print scales
# scales = [1] * 10

threshold = 0.5

# Presentation settings
present_blank = False


def get_image(t):
    # return test_images[int(t / present_time)]
    # print t, image_ind, test_labels[image_ind]
    if present_blank:
        tmp = t / present_time
        if int(tmp) != round(tmp):
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

    # input_norm = nengo.networks.EnsembleArray(Nnorm, images_mean.shape[1],
    #                                           label='norm')
    # nengo.Connection(input_images, input_norm.input, transform=trans_std)
    # nengo.Connection(input_bias, input_norm.input, transform=trans_mean)

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
            # nengo.Connection(input_images, layer.neurons,
            #                  transform=W.T, synapse=pstc)
            # trans = np.multiply(images_std, W.T)
            # trans = images_std * W.T
            # print trans.shape
            # print W.T.shape
            # break
            # nengo.Connection(input_norm.output, layer.neurons,
            #                  transform=W.T, synapse=pstc)

            nengo.Connection(input_images, layer.neurons,
                             transform=images_std * W.T, synapse=pstc)
            nengo.Connection(input_bias, layer.neurons,
                             transform=-np.multiply(images_mean,
                                                    images_std) * W.T,
                             synapse=pstc)

            # nengo.Connection(input_images, layer.neurons,
            #                  transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp * 1000, synapse=pstc)

        layers.append(layer)

    # print data.keys()
    # print W.shape
    # print data['rec_weights'].shape
    # print Wc.shape
    # print Nclass

    # --- make cleanup
    # class_layer = nengo.networks.EnsembleArray(Nclass, Wc.shape[1],
    #                                            label='class', radius=5)
    # class_bias = nengo.Node(output=bc)
    # nengo.Connection(class_bias, class_layer.input, synapse=0)
    # nengo.Connection(layers[-1].neurons, class_layer.input,
    #                  transform=Wc.T * amp * 1000, synapse=pstc)
    # probe_class = nengo.Probe(class_layer.output, synapse=0.03)

    # test = nengo.Node(output=test_dots, size_in=n_labels)
    # nengo.Connection(class_layer.output, test)
    # probe_test = nengo.Probe(test, synapse=0.01)

    # # --- make test_ensemble
    # ens_200 = nengo.Ensemble(1, len(layers[-1].neurons),
    #                          neuron_type=nengo.Direct())
    # nengo.Connection(layers[-1].neurons, ens_200, synapse=None)
    # probe_ens_200 = nengo.Probe(ens_200, synapse=0.01)

    # for i in range(Wc.shape[1]):
    #     ens = nengo.Ensemble(1, len(class_layer.ensembles[i].neurons),
    #                          neuron_type=nengo.Direct())
    #     nengo.Connection(class_layer.ensembles[i].neurons, ens, synapse=None)
    #     probe = nengo.Probe(ens, synapse=0.01)
    #     probe_ens_10.append(probe)

    am = AM(means, [[1] for i in range(num_classes)],
            input_scale=scales,
            output_utilities=True, output_thresholded_utilities=True,
            wta_output=True, wta_inhibit_scale=3.0, threshold=threshold)
    nengo.Connection(layers[-1].neurons, am.input)

    am2 = AM(means, [[1] for i in range(num_classes)],
             input_scale=scales,
             output_utilities=True, output_thresholded_utilities=True,
             neuron_type=nengo.Direct(), threshold=threshold)
    nengo.Connection(layers[-1].neurons, am2.input)

    # --- add biases to cleanup?
    bias = nengo.Node(output=1)
    am.add_input('bias', [[1]] * num_classes, input_scale=bc)
    am2.add_input('bias', [[1]] * num_classes, input_scale=bc)
    nengo.Connection(bias, am.bias)
    nengo.Connection(bias, am2.bias)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    probe_am_tu = nengo.Probe(am.thresholded_utilities, synapse=0.03)
    probe_am2_u = nengo.Probe(am2.utilities, synapse=0.03)
    probe_am2_tu = nengo.Probe(am2.thresholded_utilities, synapse=0.03)

sim = nengo.Simulator(model)

# --- simulation
for n in range(num_digits):
    print "Processing digit %i of %i" % (n, len(test_labels))
    image_ind = n
    sim.run(present_time)
    # sp_200.append(sim.data[probe_ens_200][-1])
    # for i in range(Wc.shape[1]):
    #     sp_10[i].append(sim.data[probe_ens_10[i]][-1])
    # sim.reset()

# --- Plots
from nengo.utils.matplotlib import rasterplot

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
r, c = 5, 1

plt.subplot(r, c, 1)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(r, c, 2)
rasterplot(t, sim.data[probe_layers[0]][:, :200])
plot_bars()
plt.xticks([])
plt.xlim([t[0], t[-1]])
plt.ylabel('layer 1 (500)')

plt.subplot(r, c, 3)
rasterplot(t, sim.data[probe_layers[1]])
plt.xticks([])
plt.yticks(np.linspace(0, 200, 5))
plot_bars()
plt.xlim([t[0], t[-1]])
plt.ylabel('layer 2 (200)')

colormap = plt.cm.gist_ncar
plt.subplot(r, c, 4)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
for i in range(num_classes):
    plt.plot(t, sim.data[probe_am_tu][:, i])
plt.xlim([t[0], t[-1]])
plt.ylabel('am u')

plt.subplot(r, c, 5)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(means))])
for i in range(num_classes):
    plt.plot(t, sim.data[probe_am2_u][:, i])
plt.xlim([t[0], t[-1]])
plt.ylabel('am ttu')

plt.tight_layout()
# plt.show()
plt.savefig('test_am.png')

import os
os.system('test_am.png')
