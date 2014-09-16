"""
Training an autoencoder with LIF-likes
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'mode=DEBUG_MODE'
import theano
import theano.tensor as tt

import plotting

import autoencoder
reload(autoencoder)
from autoencoder import (rms, mnist, show_recons,
                         FileObject, Autoencoder, DeepAutoencoder)

plt.ion()


def nlif(x):
    dtype = theano.config.floatX
    sigma = tt.cast(0.05, dtype=dtype)
    tau_ref = tt.cast(0.002, dtype=dtype)
    tau_rc = tt.cast(0.02, dtype=dtype)
    alpha = tt.cast(1, dtype=dtype)
    beta = tt.cast(1, dtype=dtype)  # so that f(0) = firing threshold
    amp = tt.cast(1. / 63.04, dtype=dtype)  # so that f(1) = 1

    j = alpha * x + beta - 1
    j = sigma * tt.log1p(tt.exp(j / sigma))
    v = amp / (tau_ref + tau_rc * tt.log1p(1. / j))
    return tt.switch(j > 0, v, 0.0)


# --- load the data
train, valid, test = mnist()
train_images, _ = train
valid_images, _ = valid
test_images, _ = test

for images in [train_images, valid_images, test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# --- pretrain with SGD backprop
shapes = [(28, 28), 500, 200]
funcs = [None, nlif, nlif]
rf_shapes = [(9, 9), None]
rates = [1., 1.]

n_layers = len(shapes) - 1
assert len(funcs) == len(shapes)
assert len(rf_shapes) == n_layers
assert len(rates) == n_layers

n_epochs = 15
batch_size = 100

deep = DeepAutoencoder()
data = train_images
for i in range(n_layers):
    savename = "lif-auto-%d.npz" % i
    if not os.path.exists(savename):
        auto = Autoencoder(
            shapes[i], shapes[i+1], rf_shape=rf_shapes[i],
            vis_func=funcs[i], hid_func=funcs[i+1])
        deep.autos.append(auto)
        auto.auto_sgd(data, deep, test_images,
                      n_epochs=n_epochs, rate=rates[i])
        auto.to_file(savename)
    else:
        auto = FileObject.from_file(savename)
        assert type(auto) is Autoencoder
        deep.autos.append(auto)

    data = auto.encode(data)

plt.figure(99)
plt.clf()
recons = deep.reconstruct(test_images)
show_recons(test_images, recons)

print "recons error", rms(test_images - recons, axis=1).mean()

# deep.auto_sgd(train_images, test_images, rate=0.3, n_epochs=30)
# print "recons error", rms(test_images - recons, axis=1).mean()

# --- train classifier with backprop
savename = "lif-classifier-hinge.npz"
if not os.path.exists(savename):
    deep.train_classifier(train, test)
    np.savez(savename, W=deep.W, b=deep.b)
else:
    savedata = np.load(savename)
    deep.W, deep.b = savedata['W'], savedata['b']

print "mean error", deep.test(test).mean()

# --- train with backprop
if 1:
    # deep.backprop(train, test, n_epochs=100)
    # deep.sgd(train, test, n_epochs=50)

    # deep.sgd(train, test, n_epochs=5, noise=0.5, shift=True)
    # deep.backprop(train, test, n_epochs=50, noise=0.5, shift=True)

    deep.sgd(train, test, n_epochs=50, tradeoff=1, noise=0.3, shift=True)
    print "mean error", deep.test(test).mean()

# --- try to get autoencoder back
if 0:
    deep.auto_sgd_down(train_images, test_images, rate=0.6, n_epochs=30)
    print "recons error", rms(test_images - recons, axis=1).mean()

if 0:
    # Try to learn linear reconstructor (doesn't work too well)
    import nengo

    codes = deep.encode(train_images)
    decoders, info = nengo.decoders.LstsqL2()(codes, train_images)
    print info['rmses'].mean()

    recons = np.dot(codes, decoders)
    print rms(train_images - recons, axis=1).mean()

    plt.figure(99)
    plt.clf()
    show_recons(test_images, recons)

if 0:
    # save parameters
    d = {}
    d['weights'] = [auto.W.get_value() for auto in deep.autos]
    d['biases'] = [auto.c.get_value() for auto in deep.autos]
    if all(hasattr(auto, 'V') for auto in deep.autos):
        d['rec_weights'] = [auto.V.get_value() for auto in deep.autos]
        d['rec_biases'] = [auto.b.get_value() for auto in deep.autos]
    d['Wc'] = deep.W
    d['bc'] = deep.b
    np.savez('lif-126-error.npz', **d)

if 0:
    # compute top layers mean and std
    codes = deep.encode(train_images)
    classes = np.dot(codes, deep.W) + deep.b

    plt.figure(108)
    plt.clf()
    plt.subplot(211)
    plt.hist(codes.flatten(), 100)
    plt.subplot(212)
    plt.hist(classes.flatten(), 100)

    print "code (mean, std):", codes.mean(), codes.std()
    print "class (mean, std):", classes.mean(), classes.std()
