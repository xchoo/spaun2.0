import numpy as np
import nengo

from ...configurator import cfg
from ...vocabulator import vocab


def Free_Recall_Network(item_vocab, pos_vocab, mtr_vocab, net=None,
                        net_label='FREE RECALL'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Recalled POS MB -----------------------------
        # Increase the accumulator radius to account for increased magnitude
        # of added position vectors
        acc_radius = cfg.enc_mb_acc_radius_scale * cfg.get_optimal_sp_radius()

        net.pos_recall_mb = cfg.make_mem_block(vocab=pos_vocab, reset_key=0,
                                               radius=acc_radius,
                                               n_neurons=100)
        nengo.Connection(net.pos_recall_mb.output, net.pos_recall_mb.input)

        # ------------------------- FR CConv Unit -----------------------------
        net.fr_dcconv = cfg.make_cir_conv(invert_b=True,
                                          input_magnitude=cfg.dcconv_radius)
        nengo.Connection(net.pos_recall_mb.output, net.fr_dcconv.B,
                         transform=-1.5)

        # -------------------------- FR Assoc Mem -----------------------------
        net.fr_am = \
            cfg.make_assoc_mem(item_vocab.vectors, item_vocab.vectors,
                               inhibitable=True,
                               threshold=cfg.dec_fr_min_thresh,
                               default_output_vector=np.zeros(vocab.sp_dim))
        net.fr_am.add_output_mapping('mtr_output', mtr_vocab.vectors)
        net.fr_am.add_default_output_vector(np.zeros(vocab.mtr_dim),
                                            output_name='mtr_output')

        nengo.Connection(net.fr_dcconv.output, net.fr_am.input, synapse=0.01)

        # -------------- Thresholded ensemble for FR inhibit ------------------
        fr_inhibit = cfg.make_thresh_ens_net()
        nengo.Connection(fr_inhibit.output, net.fr_am.inhibit, transform=3)

        # ----------------------- Inputs and Outputs --------------------------
        net.items_input = net.fr_dcconv.A

        net.pos_input = nengo.Node(size_in=vocab.sp_dim)
        nengo.Connection(net.pos_input, net.pos_recall_mb.input, synapse=None)

        net.pos_acc_input = net.fr_dcconv.B

        # DEBUG
        # net.pos_input = net.fr_dcconv.B
        # nengo.Connection(net.pos_input, net.pos_recall_mb.input, synapse=None)

        # net.pos_acc_input = nengo.Node(size_in=vocab.sp_dim)
        # DEBUG

        net.output = net.fr_am.output
        net.mtr_output = net.fr_am.mtr_output

        net.reset = net.pos_recall_mb.reset
        net.gate = net.pos_recall_mb.gate
        net.inhibit = fr_inhibit.input

        net.dec_failure = net.fr_am.output_default_ens

    return net
