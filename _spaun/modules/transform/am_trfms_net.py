import nengo

from ...config import cfg
from ...vocabs import item_vocab, pos_vocab, pos1_vocab


def Assoc_Mem_Transforms_Network(net=None, net_label='AM TRANSFORMS'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Inputs and Outputs --------------------------
        # NOTE: Additional nodes here for future implementation of selectable
        #       inputs to the AM's
        # TODO: Make routing to these non-hardcoded to MB2?

        net.frm_mb1 = nengo.Node(size_in=cfg.sp_dim)
        net.frm_mb2 = nengo.Node(size_in=cfg.sp_dim)
        net.frm_mb3 = nengo.Node(size_in=cfg.sp_dim)

        net.frm_cconv = nengo.Node(size_in=cfg.sp_dim)

        # --------------------- Associative Memoires --------------------------
        net.am_p1 = cfg.make_assoc_mem(
            pos1_vocab.vectors[1:cfg.max_enum_list_pos + 1, :],
            pos_vocab.vectors)
        net.am_n1 = cfg.make_assoc_mem(pos1_vocab.vectors, item_vocab.vectors)

        net.am_p2 = cfg.make_assoc_mem(
            pos_vocab.vectors,
            pos1_vocab.vectors[1:cfg.max_enum_list_pos + 1, :])
        net.am_n2 = cfg.make_assoc_mem(item_vocab.vectors, pos1_vocab.vectors)

        nengo.Connection(net.frm_mb2, net.am_p1.input, synapse=None)
        nengo.Connection(net.frm_mb2, net.am_n1.input, synapse=None)
        nengo.Connection(net.frm_cconv, net.am_p2.input, synapse=None)
        nengo.Connection(net.frm_cconv, net.am_n2.input, synapse=None)

        # ----------------------- Inputs and Outputs --------------------------
        net.pos1_to_pos = net.am_p1.output
        net.pos_to_pos1 = net.am_p2.output
        net.pos1_to_num = net.am_n1.output
        net.num_to_pos1 = net.am_n2.output

    return net
