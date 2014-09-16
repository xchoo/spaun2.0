"""
Find good neuron parameters for computing a sigmoid.
"""

import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.utils.distributions import Uniform, UniformHypersphere

N = 3
radius = 5


def sigmoid_radius(x):
    return 1. / (1 + np.exp(-radius * x))


def encoders_rates_intercepts(seed):
    rng = np.random.RandomState(seed)
    # encoders = UniformHypersphere(1, surface=True).sample(N, rng)
    # intercepts = Uniform(-1, 1).sample(N, rng)
    encoders = np.ones((N, 1))
    intercepts = Uniform(-0.5, 0.8).sample(N, rng=rng)
    max_rates = Uniform(200, 400).sample(N, rng=rng)
    return encoders, max_rates, intercepts


def residual(encoders, max_rates, intercepts, eval_points, show=False):
    radius = 5
    neurons = nengo.LIF(N)
    gains, biases = neurons.gain_bias(max_rates, intercepts)
    A = neurons.rates(np.dot(eval_points, encoders.T), gains, biases)
    y = sigmoid_radius(eval_points)
    d, _ = nengo.decoders.LstsqL2()(A, y)
    r = np.dot(A, d) - y
    r2 = np.sqrt(np.dot(r.T, r))

    if show:
        plt.figure(101)
        plt.clf()
        x = np.linspace(-1, 1, 501).reshape(-1, 1)
        a = neurons.rates(np.dot(x, encoders.T), gains, biases)
        y = sigmoid_radius(x)
        yhat = np.dot(a, d)
        plt.plot(x, y, 'k--')
        plt.plot(x, yhat)

    return r2


def find_params(savefile=None, show=False):
    rng = np.random.RandomState(9)
    eval_points = UniformHypersphere().sample(750, 1, rng=rng)

    residuals = []
    for i in range(1000):
        encoders, max_rates, intercepts = encoders_rates_intercepts(i)
        r = residual(encoders, max_rates, intercepts, eval_points)
        residuals.append((i, r))

    residuals = sorted(residuals, key=lambda x: x[1])

    seed = residuals[0][0]
    encoders, max_rates, intercepts = encoders_rates_intercepts(seed)
    residual(encoders, max_rates, intercepts, eval_points, show=show)

    if savefile:
        np.savez(savefile,
                 N=N, radius=radius, encoders=encoders,
                 max_rates=max_rates, intercepts=intercepts)

    return N, radius, encoders, max_rates, intercepts
