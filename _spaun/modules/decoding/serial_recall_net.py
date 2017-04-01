import numpy as np
import nengo

from ...configurator import cfg


def Serial_Recall_Network(item_vocab, mtr_vocab,
                          net=None, net_label='SER RECALL'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        bias_node = nengo.Node(output=1)

        # -------------------------- MB * POS~ --------------------------------
        net.item_dcconv = cfg.make_cir_conv(invert_b=True,
                                            input_magnitude=cfg.dcconv_radius)

        # ----------------- AM (Strongest component) Decoding -----------------
        net.dec_am1 = \
            cfg.make_assoc_mem(item_vocab.vectors, mtr_vocab.vectors,
                               inhibitable=True,
                               threshold=cfg.dec_am_min_thresh,
                               default_output_vector=(
                                   np.zeros(mtr_vocab.dimensions)))
        net.dec_am1.add_output_mapping(
            'linear_output', np.eye(len(item_vocab.keys)),
            net.dec_am1.threshold_shifted_linear_funcs())

        nengo.Connection(net.item_dcconv.output, net.dec_am1.input,
                         synapse=0.01)

        # ------------- AM (2nd Strongest component) Decoding -----------------
        net.dec_am2 = cfg.make_assoc_mem(item_vocab.vectors,
                                         item_vocab.vectors,
                                         inhibitable=True,
                                         threshold=0.0)
        net.dec_am2.add_output_mapping(
            'linear_output', np.eye(len(item_vocab.keys)),
            net.dec_am2.threshold_shifted_linear_funcs())

        nengo.Connection(net.item_dcconv.output, net.dec_am2.input,
                         synapse=0.01)

        # Inhibit the am1 chosen item (so that am2 chooses the 2nd strongest)
        net.dec_am2.add_input_mapping('dec_am_utils',
                                      np.eye(len(item_vocab.keys)) * -3)
        nengo.Connection(net.dec_am1.cleaned_output_utilities,
                         net.dec_am2.dec_am_utils)

        # -------------- AM Utilities Difference Calculation ------------------
        # Util diff calculation: High if difference between am1 utils and
        # am2 utils is greater than cfg.dec_am_min_diff
        util_diff = cfg.make_thresh_ens_net(cfg.dec_am_min_diff)
        nengo.Connection(net.dec_am1.linear_output, util_diff.input,
                         transform=[[1] * len(item_vocab.keys)], synapse=0.02)
        nengo.Connection(net.dec_am2.linear_output, util_diff.input,
                         transform=[[-1] * len(item_vocab.keys)], synapse=0.02)

        util_diff_neg = cfg.make_thresh_ens_net(1 - cfg.dec_am_min_diff)
        nengo.Connection(bias_node, util_diff_neg.input)
        nengo.Connection(util_diff.output, util_diff_neg.input, transform=-2,
                         synapse=0.01)
        nengo.Connection(net.dec_am1.inhibit, util_diff_neg.input,
                         transform=-2)  # WHY IS THIS HERE?

        util_diff_thresh = cfg.make_thresh_ens_net()   # Clean util_diff signal
        nengo.Connection(bias_node, util_diff_thresh.input)
        nengo.Connection(util_diff_neg.output, util_diff_thresh.input,
                         transform=-2, synapse=0.01)

        # ----------------------- Inputs and Outputs --------------------------
        net.items_input = net.item_dcconv.A
        net.pos_input = net.item_dcconv.B

        net.am_input = net.dec_am1.input
        net.dec_success = net.dec_am1.cleaned_output_utilities
        net.dec_failure = net.dec_am1.output_default_ens

        net.am_utils_diff = util_diff_thresh.output
        net.output = net.dec_am1.output

        # net.dec_am1.add_output('item_sp_out', item_vocab.vectors)
        # net.item_output = net.dec_am1.item_sp_out

        net.inhibit = nengo.Node(size_in=1)

        # ----------------------- Misc Connections ----------------------------
        nengo.Connection(net.inhibit, net.dec_am1.inhibit, transform=3,
                         synapse=None)
        nengo.Connection(net.inhibit, net.dec_am2.inhibit, transform=3,
                         synapse=None)

    return net
