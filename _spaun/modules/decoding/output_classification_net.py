import numpy as np
import nengo

from ...config import cfg


def Output_Classification_Network(net=None, net_label='OUT CLASSIFICATION'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        bias_node = nengo.Node(output=1)

        # ---------------------- Utility value Inputs -------------------------
        net.sr_utils_y = nengo.Node(size_in=1)
        net.sr_utils_n = nengo.Node(size_in=1)

        net.sr_utils_diff = nengo.Node(size_in=1)

        net.fr_utils_n = nengo.Node(size_in=1)

        net.output_unk_inhibit = nengo.Node(size_in=1)
        net.output_stop_inhibit = nengo.Node(size_in=1)

        # -------------------- Threshold circuit logic ------------------------
        # Output classification (know, don't know, list end) stuff
        # - Logic:
        #     - KNOW if
        #         (sr_am_utils > cfg.dec_am_min_thresh &&
        #          sr_am_utils_diff > cfg.dec_am_min_diff)
        #     - DON'T KNOW if
        #         ((fr_am_utils > cfg.dec_fr_min_thresh &&
        #           sr_am_utils < cfg.dec_am_min_thresh) ||
        #          (sr_am_utils_diff < cfg.dec_am_min_diff &&
        #           sr_am_utils > cfg.dec_am_min_thresh))
        #     - STOP if
        #         (sr_am_utils < cfg.dec_am_min_thresh &&
        #          fr_am_utils < cfg.dec_fr_min_thresh)

        output_know = cfg.make_thresh_ens_net(0.55)
        output_unk = cfg.make_thresh_ens_net(0.80)
        output_stop = cfg.make_thresh_ens_net(0.75)

        nengo.Connection(net.sr_utils_y, output_know.input, transform=0.5,
                         synapse=None)
        nengo.Connection(net.sr_utils_diff, output_know.input, transform=0.5,
                         synapse=None)

        nengo.Connection(bias_node, output_unk.input)
        nengo.Connection(output_know.output, output_unk.input, transform=-5,
                         synapse=0.03)
        nengo.Connection(net.output_unk_inhibit, output_unk.input,
                         transform=-2, synapse=None)

        nengo.Connection(net.sr_utils_n, output_stop.input, transform=0.5,
                         synapse=None)
        nengo.Connection(net.fr_utils_n, output_stop.input, transform=0.5,
                         synapse=None)
        nengo.Connection(net.output_stop_inhibit, output_stop.input,
                         transform=-2, synapse=None)

        # ----------------------- Inputs and Outputs --------------------------
        net.item_input = nengo.Node(cfg.sp_dim)
        net.item_output = nengo.Node(cfg.sp_dim)

        net.output_know = output_know.output
        net.output_unk = output_unk.output
        net.output_stop = output_stop.output

    return net
