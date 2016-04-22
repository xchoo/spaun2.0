import numpy as np


def get_optimal_radius(dimensions, subdimensions=1):
    """
    Returns the optimal radius to use given the dimension of the semantic
    pointer, and the dimensionailty of the sub-ensembles used in the
    EnsembleArray or SPAEnsembleArray network. The optimal radius value has
    been optimized to represent 99.95% of the range of possible vector values
    in a semantic pointer of a given dimensionality.

    Note: Optimized for dimensions < 1024

    Parameters
    ----------
    dimensions: int
        Number of dimensions of the semantic pointer.
    subdimensions: int
        Number of dimensions of the sub-ensembles used in the EnsembleArray or
        SPAEnsembleArray network.
    """

    if dimensions < 1 or subdimensions < 1:
        raise ValueError('get_optimal_radius: Invalid dimensions or ' +
                         'subdimensions values given.')

    # For dimensions < 8, the optimal radius is pretty close to 1 regardless
    # of the value of the subdimensions
    if dimensions < 8:
        return 1

    # Quadratic function used to approximate the parameters needed for the
    # exponential function below.
    def quad_func(x, a, b, c, d, e):
        return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4

    # Exponential function used to approximate the parameters of the scaling
    # factor to use when calculating the optimal radius, where:
    #
    # opt_radius = scaling_factor * sqrt(subdimensions / dimensions)
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Get the exponential function parameters. The parameters for the quadratic
    # approximation function are analytically derived.
    exp_a = quad_func(np.log(dimensions), 1.64204432, 0.82220578, -0.58990055,
                      0.10252279, -0.00565123)
    exp_b = quad_func(np.log(dimensions), 0.65365165, -0.31349688, 0.10306855,
                      -0.01210683, 0.00046143)
    exp_c = quad_func(np.log(dimensions), -0.63226615, -0.87264469, 0.62154834,
                      -0.10811827, 0.00595809)

    # Calculate the scaling factor using the exponential approximation
    # function
    scale_factor = exp_func(np.log(1.0 * dimensions / subdimensions),
                            exp_a, exp_b, exp_c)

    # Calculate the optimal radius (cap radius at 0.1 -- equiv dim = 1024)
    radius = max(0.1, scale_factor * np.sqrt(1.0 * subdimensions / dimensions))

    return radius
