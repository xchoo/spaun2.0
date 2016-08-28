import nengo

from ...configurator import cfg


def Settable_Pos_Inc_Network(pos_vocab, pos_reset_key, inc_sp, num_vocab,
                             net=None, net_label='POS INC', **args):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        num_pos_vecs = pos_vocab.vectors.shape[0]

        # Number to position sp assoc memory
        net.num_2_pos_am = \
            cfg.make_assoc_mem(num_vocab.vectors[:num_pos_vecs, :],
                               pos_vocab.vectors, threshold=0.35,
                               inhibitable=True)

        # Memory block to store POS vector
        net.pos_mb = cfg.make_mem_block(
            label="POS MB", vocab=pos_vocab, reset_key=pos_reset_key,
            cleanup_mode=cfg.instr_pos_inc_cleanup_mode, **args)
        nengo.Connection(net.num_2_pos_am.output, net.pos_mb.input)

        # POS x INC
        nengo.Connection(net.pos_mb.output, net.pos_mb.input,
                         transform=inc_sp.get_convolution_matrix())

        # Disable pos_mb mem2 output when am is active
        nengo.Connection(net.num_2_pos_am.output_utilities,
                         net.pos_mb.mem2.reset,
                         transform=[[1] * num_pos_vecs])

        # Output node
        # -- Combine the output of the num_2_pos_am and the pos_mb (which is
        #    disabled when there is a valid output from the am)
        net.output = nengo.Node(size_in=pos_vocab.dimensions)
        nengo.Connection(net.pos_mb.output, net.output)
        nengo.Connection(net.num_2_pos_am.output, net.output)

        net.inhibit_am = net.num_2_pos_am.inhibit
        net.input = net.num_2_pos_am.input
        net.reset = net.pos_mb.reset
        net.gate = net.pos_mb.gate

        # #### DEBUG ####
        net.input_debug = nengo.Node(size_in=num_pos_vecs)
        nengo.Connection(net.input, net.input_debug,
                         transform=num_vocab.vectors[:num_pos_vecs, :])
    return net
