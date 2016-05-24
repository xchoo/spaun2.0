import numpy as np

import nengo
from nengo.networks import EnsembleArray

from ..._networks import convert_func_2_diff_func
from ...configurator import cfg


def Visual_Transform_Network(vis_vocab, vis_am_threshold, vis_am_input_scale,
                             copy_draw_trfms_x, copy_draw_trfms_y,
                             mtr_vocab, mtr_sp_scale_factor,
                             net=None, net_label='VIS TRFM'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Inputs and Outputs --------------------------
        net.input = nengo.Node(size_in=vis_vocab.dimensions)
        net.output = nengo.Node(size_in=mtr_vocab.dimensions)

        # ------------------ Digit (Answer) Classification --------------------
        # Takes digit semantic pointer from visual wm and identifies
        # appropriate digit classification
        # - Generates: Digit class (I - 1) used for inhibition.
        # -            Default output vectors inhibits all.
        # Note: threshold is halved to compenstate (sort of) for drift in the
        #       visual WM system
        digit_classify = \
            cfg.make_assoc_mem(vis_vocab.vectors[:len(mtr_vocab.keys), :],
                               np.ones((len(mtr_vocab.keys),
                                        len(mtr_vocab.keys))) -
                               np.eye(len(mtr_vocab.keys)),
                               threshold=vis_am_threshold,
                               label='DIGIT CLASSIFY')
        digit_classify.add_default_output_vector(np.ones(len(mtr_vocab.keys)))
        nengo.Connection(net.input, digit_classify.input,
                         transform=vis_am_input_scale, synapse=None)

        # --------------------- Motor SP Transformation -----------------------
        if len(mtr_vocab.keys) != copy_draw_trfms_x.shape[0]:
            raise ValueError('Transform System - Number of motor pointers',
                             ' does not match number of given copydraw',
                             ' transforms.')

        # ------------------ Motor SP Transform ensembles ---------------------
        for n in range(len(mtr_vocab.keys)):
            mtr_path_dim = mtr_vocab.dimensions // 2
            # Motor SP contains both X and Y information, so motor path dim is
            # half that of the SP dim

            # trfm_x = convert_func_2_diff_func(copy_draw_trfms_x[n])
            # trfm_y = convert_func_2_diff_func(copy_draw_trfms_y[n])
            trfm_x = np.array(copy_draw_trfms_x[n])
            trfm_y = np.array(copy_draw_trfms_y[n])

            trfm_ea = EnsembleArray(n_neurons=cfg.n_neurons_ens,
                                    n_ensembles=mtr_vocab.dimensions,
                                    radius=mtr_sp_scale_factor)
            cfg.make_inhibitable(trfm_ea)

            nengo.Connection(net.input, trfm_ea.input[:mtr_path_dim],
                             transform=trfm_x.T, synapse=None)
            nengo.Connection(net.input, trfm_ea.input[mtr_path_dim:],
                             transform=trfm_y.T, synapse=None)

            # Class output is inverted (i.e. if class is 3, it's [1, 1, 0, 1])
            # So transform here is just the identity
            inhib_trfm = np.zeros((1, len(mtr_vocab.keys)))
            inhib_trfm[0, n] = 1
            nengo.Connection(digit_classify.output, trfm_ea.inhibit,
                             transform=inhib_trfm)
            nengo.Connection(trfm_ea.output, net.output, synapse=None)

    return net


def Dummy_Visual_Transform_Network(vectors_in, vectors_out, net=None,
                                   net_label='DUMMY VIS TRFM'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ------------------ Digit (Answer) Classification --------------------
        # Takes digit semantic pointer from visual wm and identifies
        # appropriate digit classification
        # - Generates: Digit class (I - 1) used for inhibition.
        # -            Default output vectors inhibits all.
        digit_classify = \
            cfg.make_assoc_mem(vectors_in, vectors_out, label='DIGIT CLASSIFY')

        # ----------------------- Inputs and Outputs --------------------------
        net.input = digit_classify.input
        net.output = digit_classify.output

    return net
