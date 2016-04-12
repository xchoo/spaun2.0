import numpy as np
import nengo
import os

from ..._networks import convert_func_2_diff_func
from ...config import cfg
from ...vocabs import mtr_vocab, mtr_filepath, mtr_sp_scale_factor
from ..vision.lif_vision import am_threshold, am_vis_sps
from ..vision.lif_vision import max_rate as lif_vis_max_rate


def Visual_Transform_Network(net=None, net_label='VIS TRFM'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Inputs and Outputs --------------------------
        net.input = nengo.Node(size_in=cfg.vis_dim)
        net.output = nengo.Node(size_in=cfg.mtr_dim)

        # ------------------ Digit (Answer) Classification --------------------
        # Takes digit semantic pointer from visual wm and identifies
        # appropriate digit classification
        # - Generates: Digit class (I - 1) used for inhibition.
        # -            Default output vectors inhibits all.
        digit_classify = \
            cfg.make_assoc_mem(am_vis_sps[:len(mtr_vocab.keys), :],
                               np.ones((len(mtr_vocab.keys),
                                        len(mtr_vocab.keys))) -
                               np.eye(len(mtr_vocab.keys)),
                               threshold=am_threshold, label='DIGIT CLASSIFY')
        digit_classify.add_default_output_vector(np.ones(len(mtr_vocab.keys)))
        nengo.Connection(net.input, digit_classify.input,
                         transform=lif_vis_max_rate)

        # --------------------- Motor SP Transformation -----------------------
        # Takes visual SP and transforms them to the 'copy-drawn' motor SP
        copy_draw_tfrm_data = np.load(os.path.join(mtr_filepath,
                                                   'copydraw_trfms.npz'))
        copy_draw_trfms_x = copy_draw_tfrm_data['trfms_x']
        copy_draw_trfms_y = copy_draw_tfrm_data['trfms_y']

        if len(mtr_vocab.keys) != copy_draw_trfms_x.shape[0]:
            raise ValueError('Transform System - Number of motor pointers',
                             ' does not match number of given copydraw',
                             ' transforms.')

        # ------------------ Motor SP Transform ensembles ---------------------
        for n in range(len(mtr_vocab.keys)):
            mtr_path_dim = cfg.mtr_dim // 2
            # Motor SP contains both X and Y information, so motor path dim is
            # half that of the SP dim

            trfm_x = convert_func_2_diff_func(copy_draw_trfms_x[n])
            trfm_y = convert_func_2_diff_func(copy_draw_trfms_y[n])

            trfm_ea = cfg.make_ens_array(n_ensembles=cfg.mtr_dim,
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
            nengo.Connection(digit_classify.output, trfm_ea.inhib,
                             transform=inhib_trfm)
            nengo.Connection(trfm_ea.output, net.output, synapse=None)

    return net
