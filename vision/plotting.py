"""
Helper functions for plotting when training autoencoders and RBMs.
"""

import os

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

def display_available():
    return ('DISPLAY' in os.environ)


def show(image, ax=None, vlims=None, invert=False):
    kwargs = dict(interpolation='none')
    if image.ndim == 2:
        kwargs['cmap'] = 'gray' if not invert else 'gist_yarg'
        if vlims is not None:
            kwargs['vmin'], kwargs['vmax'] = vlims
    elif image.ndim == 3:
        assert image.shape[2] == 3
        if vlims is not None:
            image = (image.clip(*vlims) - vlims[0]) / (vlims[1] - vlims[0])
    else:
        raise ValueError("Wrong number of image dimensions")

    if ax is None: ax = plt.gca()
    ax.imshow(image, **kwargs)
    return ax


def tile(images, ax=None, rows=16, cols=24, random=False,
         grid=False, gridwidth=1, gridcolor='r', **show_params):
    """
    Plot tiled images to the current axis

    :images Each row is one flattened image
    """

    n_images = images.shape[0]
    imshape = images.shape[1:]
    m, n = imshape[:2]
    n_channels = imshape[2] if len(imshape) > 2 else 1

    inds = np.arange(n_images)
    if random:
        npr.shuffle(inds)

    img_shape = (m*rows, n*cols)
    if n_channels > 1:
        img_shape = img_shape + (n_channels,)
    img = np.zeros(img_shape, dtype=images.dtype)
    for ind in xrange(min(rows*cols, n_images)):
        i,j = (ind / cols, ind % cols)
        img[i*m:(i+1)*m, j*n:(j+1)*n] = images[inds[ind]]


    ax = show(img, ax=ax, **show_params)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if grid:
        for i in xrange(1,rows):
            ax.plot([-0.5, img.shape[1]-0.5], [i*m-0.5, i*m-0.5], '-',
                    color=gridcolor, linewidth=gridwidth)
        for j in xrange(1,cols):
            ax.plot([j*n-0.5, j*n-0.5], [-0.5, img.shape[0]-0.5], '-',
                    color=gridcolor, linewidth=gridwidth)

        ax.set_xlim([-0.5, img.shape[1]-0.5])
        ax.set_ylim([-0.5, img.shape[0]-0.5])
        ax.invert_yaxis()


def compare(imagesetlist,
            ax=None, rows=5, cols=20, vlims=None, grid=True, random=False):
    d = len(imagesetlist)

    n_images = imagesetlist[0].shape[0]
    imshape = imagesetlist[0].shape[1:]
    m, n = imshape[:2]
    n_channels = imshape[2] if len(imshape) > 2 else 1

    inds = np.arange(n_images)
    if random:
        npr.shuffle(inds)

    img_shape = (d*m*rows, n*cols)
    if n_channels > 1:
        img_shape = img_shape + (n_channels,)
    img = np.zeros(img_shape, dtype=imagesetlist[0].dtype)

    for ind in range(min(rows*cols, n_images)):
        i,j = (ind / cols, ind % cols)
        for k in xrange(d):
            img[(d*i+k)*m:(d*i+k+1)*m, j*n:(j+1)*n] = \
                imagesetlist[k][inds[ind],:].reshape(imshape)

    ax = show(img, ax=ax, vlims=vlims)

    if grid:
        for i in xrange(1,rows):
            ax.plot( [-0.5, img.shape[1]-0.5], (d*i*m-0.5)*np.ones(2), 'r-' )
        for j in xrange(1,cols):
            ax.plot( [j*n-0.5, j*n-0.5], [-0.5, img.shape[0]-0.5], 'r-')

        ax.set_xlim([-0.5, img.shape[1]-0.5])
        ax.set_ylim([-0.5, img.shape[0]-0.5])
        ax.invert_yaxis()


def activations(acts, func, ax=None):
    if ax is None:
        ax = plt.gca()

    N = acts.size
    nbins = max(np.sqrt(N), 10)

    minact, maxact = (-2, 2)
    ax.hist(acts.ravel(), bins=nbins, range=(minact,maxact), normed=True)

    x = np.linspace(minact, maxact, 101)
    ax.plot(x, func(x))

    ax.set_xlim([minact, maxact])


def filters(filters, ax=None, **kwargs):
    std = filters.std()
    tile(filters, ax=ax, vlims=(-2*std, 2*std), grid=True, **kwargs)
