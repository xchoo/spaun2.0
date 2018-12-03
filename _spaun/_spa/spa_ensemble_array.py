from nengo.networks import EnsembleArray

from .utils import get_optimal_radius


class SPAEnsembleArray(EnsembleArray):
    """An array of ensembles. Specifically configured to represent semantic
    pointers.

    This network is subclassed from the nengo.networks.EnsembleArray class.
    Further documentation can be obtained there.
    Parameters
    ----------
    n_neurons : int
        The number of neurons in each sub-ensemble.
    dimensions: int, optional
        The dimensionality of the semantic pointer to be represented.
        The radii of each subensemble is scaled to
    n_ensembles : int, optional
        The number of sub-ensembles to create.
        Note: dimensions / n_ensembles must be a whole number

    Note: One of dimensions or n_ensembles must be specified

    represent_identity: bool, optional
        Whether this network is capable of representing the identity semantic
        pointer ([1, 0, 0, ..., 0]). Default: False

    label : str, optional
        A name to assign this EnsembleArray.
        Used for visualization and debugging.
    seed : int, optional
        Random number seed that will be used in the build step.
    add_to_container : bool, optional
        Whether this network will be added to the current context.
    Additional parameters for each sub-ensemble can be passed through
    ``**ens_kwargs``.
    """

    def __init__(self, n_neurons, dimensions=-1, n_ensembles=-1,
                 represent_identity=False, identity_radius=1.0,
                 label=None, seed=None, add_to_container=None, **ens_kwargs):

        if dimensions < 0 and n_ensembles < 0:
            raise ValueError('SPAEnsembleArray - One of `dimensions` or ' +
                             '`n_ensembles` must be specified.')
        if dimensions < 0:
            dimensions = n_ensembles
        if n_ensembles < 0:
            n_ensembles = dimensions

        if dimensions % n_ensembles == 0:
            ens_dimensions = dimensions // n_ensembles
        else:
            raise ValueError('SPAEnsembleArray - Invalid `dimensions` or ' +
                             '`n_ensembles` value. dimensions must be wholly' +
                             ' divisable by `n_ensembles`. Given: ' +
                             '`dimensions`: %d, ' % dimensions +
                             '`n_ensembles`: %d' % n_ensembles)

        # Set radius parameter for the base ensemble array
        ens_kwargs = dict(ens_kwargs)
        ens_radius = ens_kwargs.pop('radius',
                                    get_optimal_radius(dimensions,
                                                       ens_dimensions))
        ens_kwargs['radius'] = ens_radius

        super(SPAEnsembleArray, self).__init__(n_neurons, n_ensembles,
                                               ens_dimensions,
                                               label, seed, add_to_container,
                                               **ens_kwargs)

        # If SPA ensemble array needs to represent the identity vector,
        # modify the radius of the first element ensemble to identity_radius
        if represent_identity:
            self.ensembles[0].radius = identity_radius
