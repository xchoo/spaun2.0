import nengo

from ...configurator import cfg


def Pos_Inc_Network(pos_vocab, pos_reset_key, inc_sp,
                    net=None, net_label='POS INC', **args):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # Memory block to store POS vector
        net.pos_mb = cfg.make_mem_block(label="POS MB", vocab=pos_vocab,
                                        reset_key=pos_reset_key,
                                        cleanup_mode=cfg.enc_pos_cleanup_mode,
                                        **args)

        # POS x INC
        nengo.Connection(net.pos_mb.output, net.pos_mb.input,
                         transform=inc_sp.get_convolution_matrix())

        net.reset = net.pos_mb.reset
        net.gate = net.pos_mb.gate
        net.output = net.pos_mb.output

    return net
