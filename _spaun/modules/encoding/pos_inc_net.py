import nengo

from ...config import cfg
from ...vocabs import vocab, pos_vocab
from ...vocabs import pos_sp_strs


def Pos_Inc_Network(net=None, net_label='POS INC', vocab=vocab,
                    pos_vocab=pos_vocab, pos_cleanup_keys=pos_sp_strs,
                    pos_reset_key='POS1', inc_key='INC'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # Memory block to store POS vector
        net.pos_mb = cfg.make_mem_block(label="POS MB", vocab=pos_vocab,
                                        cleanup_keys=pos_cleanup_keys,
                                        reset_key=pos_reset_key)

        # POS x INC
        nengo.Connection(net.pos_mb.output, net.pos_mb.input,
                         transform=vocab[inc_key].get_convolution_matrix())

        net.reset = net.pos_mb.reset
        net.gate = net.pos_mb.gate
        net.output = net.pos_mb.output

    return net
