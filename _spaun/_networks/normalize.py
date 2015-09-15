import numpy as np

import nengo
from nengo.networks import EnsembleArray

from . import Product


def scale_trfm(min_in, max_in, min_out=-1, max_out=1):
    in_range = (max_in - min_in) * 1.0
    out_range = (max_out - min_out) * 1.0
    return out_range / in_range


def scale_bias(min_in, max_in, min_out=-1, max_out=1):
    in_range = (max_in - min_in) * 1.0
    out_range = (max_out - min_out) * 1.0
    return -(min_in * out_range - min_out * in_range) / in_range


def prod_out_func(a, b, a_in_trfm, a_in_bias, b_in_trfm, b_in_bias):
    new_a = (a - a_in_bias) / a_in_trfm
    new_b = (b - b_in_bias) / b_in_trfm
    return new_a * new_b


def norm_subtract_func(x, in_trfm, in_bias, out_trfm, out_bias):
    new_x = (x - in_bias) / in_trfm
    result = 0 if new_x <= 0 else (1.0 - 1.0 / np.sqrt(new_x))
    return result * out_trfm + out_bias


def VectorNormalize(min_mag, max_mag, dimensions, radius_scale=None,
                    n_neurons_norm=50, n_neurons_norm_sub=50,
                    n_neurons_prod=150, norm_error_per_dimension=0.0003,
                    net=None):
    if net is None:
        net = nengo.Network(label="Vector Normalize")

    if radius_scale is None:
        radius_scale = 3.5 / np.sqrt(dimensions)

    max_radius_scale = max_mag * radius_scale

    norm_sub_in_low = min_mag ** 2
    norm_sub_in_high = max_mag ** 2
    norm_sub_in_trfm = scale_trfm(norm_sub_in_low, norm_sub_in_high)
    norm_sub_in_bias = scale_bias(norm_sub_in_low, norm_sub_in_high)

    norm_sub_in_bias_offset = -(norm_error_per_dimension * dimensions)

    prod_a_trfm = 1.0 / max_radius_scale
    prod_b_low = 1.0 - 1.0 / min_mag
    prod_b_high = 1.0 - 1.0 / max_mag
    prod_b_trfm = scale_trfm(prod_b_low, prod_b_high)
    prod_b_bias = scale_bias(prod_b_low, prod_b_high)

    norm_sub_func = lambda x: norm_subtract_func(x, norm_sub_in_trfm,
                                                 norm_sub_in_bias,
                                                 prod_b_trfm, prod_b_bias)
    prod_func = lambda x, y: prod_out_func(x, y, prod_a_trfm, 0,
                                           prod_b_trfm, prod_b_bias)

    with net:
        net.input = nengo.Node(size_in=dimensions)
        net.output = nengo.Node(size_in=dimensions)
        bias_node = nengo.Node(1)

        # Ensemble array to represent input vector and to compute vector
        # norm
        norm_array = EnsembleArray(n_neurons_norm, dimensions,
                                   radius=max_radius_scale)
        norm_array.add_output('squared', lambda x: x ** 2)
        nengo.Connection(net.input, norm_array.input)

        # Ensemble to calculate amount of magnitude to be subtracted
        # i.e. (1 - 1 / np.linalg.norm(input))
        norm_subtract_ens = nengo.Ensemble(50, 1, n_eval_points=5000)
        nengo.Connection(norm_array.squared, norm_subtract_ens,
                         transform=np.ones((1, dimensions)) * norm_sub_in_trfm)
        nengo.Connection(bias_node, norm_subtract_ens,
                         transform=norm_sub_in_bias + norm_sub_in_bias_offset)

        # Product network to compute product between input vector and
        # magnitude to be subtracted
        prod_array = Product(n_neurons_prod, dimensions)
        for e in prod_array.product.ensembles:
            e.n_eval_points = 5000
        prod_array.product.add_output('prod2', lambda x: prod_func(x[0], x[1]))
        prod_array.prod2 = prod_array.product.prod2

        nengo.Connection(norm_array.output, prod_array.A,
                         transform=prod_a_trfm)
        nengo.Connection(norm_subtract_ens, prod_array.B,
                         function=norm_sub_func,
                         transform=np.ones((dimensions, 1)))

        # Output connections
        nengo.Connection(norm_array.output, net.output)
        nengo.Connection(prod_array.prod2, net.output, transform=-1)
    return net
