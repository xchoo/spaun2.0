import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import nengo

# --- parameters
present_time = 0.1
num_digits = 15
# Ncode = 10
# Nclass = 30
Nclass = 50
# pstc = 0.006
pstc = 0.005

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
# _, _, [test_images, test_labels] = mnist('spaun_sym.pkl.gz')
_, _, [test_images, test_labels] = mnist()

# --- Normalize images
# for images in [test_images]:
#     images -= images.mean(axis=0, keepdims=True)
#     images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# --- Shuffle images
rng = np.random.RandomState(None)
inds = rng.permutation(len(test_images))
sorted_inds = np.argsort(test_labels)
test_images = test_images[sorted_inds]
test_labels = test_labels[sorted_inds]
num_digits = len(test_labels)

# --- SP data
sp_200 = []
sp_10 = [[] for _ in range(Wc.shape[1])]
probe_ens_10 = []

# --- Save image data
np.savez('mnist.npz', images=test_images, labels=test_labels)

labels = np.unique(test_labels)
n_labels = labels.size

def get_image(t):
    # return test_images[int(t / present_time)]
    # print t, image_ind, test_labels[image_ind]
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
        layer = nengo.Ensemble(n, 1, label='layer %d' % i,
                               neuron_type=neuron_type,
                               max_rates=max_rate * np.ones(n),
                               intercepts=intercept * np.ones(n))
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.neurons, transform=np.eye(n), synapse=0)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp * 1000, synapse=pstc)

        layers.append(layer)

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(Nclass, Wc.shape[1],
                                               label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(layers[-1].neurons, class_layer.input,
                     transform=Wc.T * amp * 1000, synapse=pstc)

    # test = nengo.Node(output=test_dots, size_in=n_labels)
    # nengo.Connection(class_layer.output, test)

    # --- make test_ensemble
    ens_200 = nengo.Ensemble(1, len(layers[-1].neurons),
                             neuron_type=nengo.Direct())
    nengo.Connection(layers[-1].neurons, ens_200, synapse=None)

    for i in range(Wc.shape[1]):
        ens = nengo.Ensemble(1, len(class_layer.ensembles[i].neurons),
                             neuron_type=nengo.Direct())
        nengo.Connection(class_layer.ensembles[i].neurons, ens, synapse=None)
        probe = nengo.Probe(ens, synapse=0.01)
        probe_ens_10.append(probe)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    # probe_test = nengo.Probe(test, synapse=0.01)
    probe_ens_200 = nengo.Probe(ens_200, synapse=0.01)

sim = nengo.Simulator(model)

# --- simulation
class_10 = []
for n in range(num_digits):
    print "Processing digit %i of %i" % (n, len(test_labels))
    image_ind = n
    sim.run(present_time)
    sp_200.append(sim.data[probe_ens_200][-1])
    for i in range(Wc.shape[1]):
        sp_10[i].append(sim.data[probe_ens_10[i]][-1])
    class_10.append(sim.data[probe_class][-1])
    sim.reset()

#sim.run(100.)
# sim.run(2.)

SP_kwargs = {}
SP_kwargs['200D'] = sp_200
for i in range(Wc.shape[1]):
    SP_kwargs['10D%i' % i] = sp_10[i]
np.savez('SPs_SYM.npz', **SP_kwargs)
np.savez('class_10.npz', class_10=class_10)

# test_sp = sim.data[probe_test_ens][-1]
# print len(test_sp)
# print np.linalg.norm(test_sp)

# t = sim.trange()

# --- plots
# from nengo.utils.matplotlib import rasterplot


# def plot_bars():
#     ylim = plt.ylim()
#     for x in np.arange(0, t[-1], present_time):
#         plt.plot([x, x], ylim, 'k--')

# inds = slice(0, int(t[-1] / present_time) + 1)
# images = test_images[inds]
# labels = test_labels[inds]
# allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
# for i, image in enumerate(images):
#     allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

# plt.figure(1)
# plt.clf()
# r, c = 5, 1

# plt.subplot(r, c, 1)
# plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(r, c, 2)
# rasterplot(t, sim.data[probe_layers[0]][:, :200])
# plot_bars()
# plt.xticks([])
# plt.ylabel('layer 1 (500)')

# plt.subplot(r, c, 3)
# rasterplot(t, sim.data[probe_layers[1]])
# plt.xticks([])
# plt.yticks(np.linspace(0, 200, 5))
# plot_bars()
# plt.ylabel('layer 2 (200)')

# plt.subplot(r, c, 4)
# plt.plot(t, sim.data[probe_class])
# plot_bars()
# plt.xlabel('time [s]')
# plt.ylabel('class')

# plt.subplot(r, c, 5)
# plt.plot(t, sim.data[probe_test])
# plt.ylim([-0.1, 1.1])
# plot_bars()
# plt.xlabel('time [s]')
# plt.ylabel('correct')

# plt.tight_layout()

# plt.savefig('run_lif.png')
# plt.show()

# --- compute error rate
# zblocks = sim.data[probe_test].reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
# errors = np.mean(zblocks, axis=1) < 0.5
# print errors.mean()

# z2 = np.argmax(y, axis=1) == labels.repeat(100)
# zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
# errors = np.mean(zblocks, axis=1) < 0.5
# print errors.mean()
