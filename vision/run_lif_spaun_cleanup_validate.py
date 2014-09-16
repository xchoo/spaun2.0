import numpy as np

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
Nnorm = 30
# pstc = 0.006
pstc = 0.005

max_rate = 63.04
intercept = 0
amp = 1. / max_rate


cleanup_threshold = 0.5  # 0.7

prenormalize_images = False
normalize_images = True
include_retina = False

normalize_images = normalize_images and not prenormalize_images
print prenormalize_images, normalize_images, include_retina

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

# ---
num_digits = len(test_images)

# --- Normalize images
images_mean = np.array([[1] * test_images[0].shape[0]])
images_std = np.array([[1] * test_images[0].shape[0]])
if prenormalize_images:
    for images in [test_images]:
        images -= images.mean(axis=0, keepdims=True)
        images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

if normalize_images:
    images_mean = test_images.mean(axis=0, keepdims=True)
    images_std = 1.0 / np.maximum(test_images.std(axis=0, keepdims=True), 3e-1)

if include_retina:
    trans_std = np.eye(images_std.shape[1]) * images_std
    trans_mean = -np.eye(images_mean.shape[1]) * np.multiply(images_mean,
                                                             images_std)

# --- Shuffle images
rng = np.random.RandomState(None)
inds = rng.permutation(len(test_images))
inds = inds[range(num_digits)]
test_images = test_images[inds]
test_labels = test_labels[inds]

# --- Load mean data
# means = np.load('means_200D.npz')['means']
means = Wc.T * amp / dt / 4  # 2
scales = np.load('scales_200D.npz')['scales']
scales = [1] * 10

# Presentation settings
present_blank = False


def get_image(t):
    tmp = t / present_time
    if int(tmp) != round(tmp) and present_blank:
        return [0] * len(test_images[image_ind])
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

    if include_retina:
        input_norm = nengo.networks.EnsembleArray(Nnorm, images_mean.shape[1],
                                                  label='norm')
        nengo.Connection(input_images, input_norm.input, transform=trans_std)
        nengo.Connection(input_bias, input_norm.input, transform=trans_mean)

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
            if prenormalize_images:
                nengo.Connection(input_images, layer.neurons,
                                 transform=W.T, synapse=pstc)

            if normalize_images:
                nengo.Connection(input_images, layer.neurons,
                                 transform=images_std * W.T, synapse=pstc)
                nengo.Connection(input_bias, layer.neurons,
                                 transform=-np.multiply(images_mean,
                                                        images_std) * W.T,
                                 synapse=pstc)

            if include_retina:
                nengo.Connection(input_norm.output, layer.neurons,
                                 transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp * 1000, synapse=pstc)

        layers.append(layer)

    # --- add cleanup
    threshold = cleanup_threshold

    am = AM(means, [[1] for i in range(len(labels))],
            input_scale=scales, n_neurons_per_ensemble=50,
            output_utilities=True, output_thresholded_utilities=True,
            wta_output=True, wta_inhibit_scale=3.0, threshold=threshold)
    nengo.Connection(layers[-1].neurons, am.input, synapse=pstc)

    # --- add biases to cleanup?
    bias = nengo.Node(output=1)
    am.add_input('bias', [[1]] * len(labels), input_scale=bc)
    nengo.Connection(bias, am.bias, synapse=pstc)

    # --- make probes
    probe_layers = [nengo.Probe(layer, 'spikes') for layer in layers]
    probe_am_tu = nengo.Probe(am.thresholded_utilities, synapse=0.03)

# -- Count all neurons
print "Total Neurons: %d" % (sum([e.n_neurons for e in model.all_ensembles]))

sim = nengo.Simulator(model)


def find_indicies(lst, value=True):
    return [i for i, j in enumerate(lst) if j == value]

results = []
# --- simulation
for n in range(num_digits):
    print "Processing digit %i of %i" % (n, len(test_labels))
    image_ind = n
    sim.run(present_time)
    data = sim.data[probe_am_tu][-1]

    net_ans = find_indicies([n > 0.3 for n in data])
    ref_ans = test_labels[image_ind]

    result = (ref_ans in net_ans) * 1.0 + (len(net_ans) > 1) * 1.0
    results.append(result)

    sim.reset()

results = np.array(results)

np.savez('validate_%d.npz' % int(rng.rand(1) * 10000),
         results=results, ind=inds, threshold=cleanup_threshold,
         normalize_images=normalize_images)

print "Correct: %f, Multi: %f, Wrong: %f" % (
    sum([n == 1 for n in results]) / (num_digits * 1.0),
    sum([n == 2 for n in results]) / (num_digits * 1.0),
    sum([n == 0 for n in results]) / (num_digits * 1.0))

# print inds
# print np.argsort(inds)
# print inds[np.argsort(inds)]
# print results[np.argsort(inds)]
