import numpy as np
import nengo

from ...config import cfg
from ...vocabs import vocab, item_vocab, mtr_vocab


def Free_Recall_Network(net=None, net_label='FREE RECALL'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Recalled POS MB -----------------------------
        net.pos_recall_mb = cfg.make_mem_block(vocab=vocab, reset_key=0)
        nengo.Connection(net.pos_recall_mb.output, net.pos_recall_mb.input,
                         transform=cfg.enc_mb_acc_fdbk_scale)

        # ------------------------- FR CConv Unit -----------------------------
        fr_dcconv = cfg.make_cir_conv(invert_b=True,
                                      input_magnitude=cfg.dcconv_radius)
        nengo.Connection(net.pos_recall_mb.output, fr_dcconv.B, transform=-1)

        # -------------------------- FR Assoc Mem -----------------------------
        net.fr_am = \
            cfg.make_assoc_mem(item_vocab.vectors, item_vocab.vectors,
                               inhibitable=True,
                               threshold=cfg.dec_fr_min_thresh,
                               default_output_vector=np.zeros(cfg.sp_dim))
        net.fr_am.add_output_mapping('mtr_output', mtr_vocab.vectors)
        net.fr_am.add_default_output_vector(np.zeros(cfg.mtr_dim),
                                            output_name='mtr_output')

        nengo.Connection(fr_dcconv.output, net.fr_am.input, synapse=0.01)

        # ----------------------- Inputs and Outputs --------------------------
        net.items_input = fr_dcconv.A

        net.pos_input = nengo.Node(size_in=cfg.sp_dim)
        nengo.Connection(net.pos_input, net.pos_recall_mb.input, synapse=None)

        net.pos_acc_input = fr_dcconv.B

        net.output = net.fr_am.output
        net.mtr_output = net.fr_am.mtr_output

        net.reset = net.pos_recall_mb.reset
        net.gate = net.pos_recall_mb.gate
        net.inhibit = nengo.Node(size_in=1)

        net.dec_failure = net.fr_am.output_default_ens

        # ----------------------- Misc Connections ----------------------------
        nengo.Connection(net.inhibit, net.fr_am.inhibit, transform=3,
                         synapse=None)

    return net
