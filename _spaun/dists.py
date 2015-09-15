import numpy as np

from nengo.dists import Distribution


class ClippedExpDist(Distribution):
    """An exponential distribution where high values are clipped.
    Parameters
    ----------
    scale : float
        The scale parameter (inverse of the rate parameter lambda).
    shift : float, optional
        Amount to shift the distribution by. There will be no values smaller
        than this shift when sampling from the distribution.
    high : float, optional
        All values larger than this value will be clipped to this value.
    """
    def __init__(self, scale, shift=0., high=np.inf):
        self.scale = scale
        self.shift = shift
        self.high = high

    def sample(self, n, d=None, rng=np.random):
        shape = (n,) if d is None else (n, d)
        exp_val = rng.exponential(self.scale, shape) + self.shift
        high = np.nextafter(self.high,
                            np.asarray(-np.inf, dtype=exp_val.dtype))
        return np.clip(exp_val, self.shift, high)
