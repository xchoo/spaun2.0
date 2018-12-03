import nengo

from ...configurator import cfg


def Assoc_Mem_Transforms_Network(item_vocab, pos_vocab, pos1_vocab,
                                 max_enum_list_pos, action_learn_vocab,
                                 cmp_vocab, net=None,
                                 net_label='AM TRANSFORMS'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        # ----------------------- Inputs and Outputs --------------------------
        # NOTE: Additional nodes here for future implementation of selectable
        #       inputs to the AM's
        # TODO: Make routing to these non-hardcoded to MB2?

        net.frm_mb1 = nengo.Node(size_in=pos1_vocab.dimensions)
        net.frm_mb2 = nengo.Node(size_in=pos1_vocab.dimensions)
        net.frm_mb3 = nengo.Node(size_in=pos1_vocab.dimensions)

        net.frm_cconv = nengo.Node(size_in=pos1_vocab.dimensions)

        net.frm_action = nengo.Node(size_in=action_learn_vocab.dimensions)
        net.action_out = nengo.Node(size_in=action_learn_vocab.dimensions)

        net.frm_compare = nengo.Node(size_in=cmp_vocab.dimensions)
        net.compare_out = nengo.Node(size_in=pos1_vocab.dimensions)

        # ---------------- Associative Memories for Q & A ---------------------
        net.am_p1 = cfg.make_assoc_mem(
            pos1_vocab.vectors[1:max_enum_list_pos + 1, :],
            pos_vocab.vectors)
        net.am_n1 = cfg.make_assoc_mem(pos1_vocab.vectors, item_vocab.vectors)

        net.am_p2 = cfg.make_assoc_mem(
            pos_vocab.vectors,
            pos1_vocab.vectors[1:max_enum_list_pos + 1, :])
        net.am_n2 = cfg.make_assoc_mem(item_vocab.vectors, pos1_vocab.vectors)

        nengo.Connection(net.frm_mb2, net.am_p1.input, synapse=None)
        nengo.Connection(net.frm_mb2, net.am_n1.input, synapse=None)
        nengo.Connection(net.frm_cconv, net.am_p2.input, synapse=None)
        nengo.Connection(net.frm_cconv, net.am_n2.input, synapse=None)

        # ----------- Associative Memories for action mapping -----------------
        # For learning task action mapping
        net.am_action_learn = cfg.make_assoc_mem(
            action_learn_vocab.vectors,
            pos1_vocab.vectors[:len(action_learn_vocab.keys), :])

        nengo.Connection(net.frm_action, net.am_action_learn.input,
                         synapse=None)
        nengo.Connection(net.am_action_learn.output, net.action_out,
                         synapse=None)

        # ------------- Associative Memories for compare task -----------------
        net.am_compare = cfg.make_assoc_mem(
            cmp_vocab.vectors,
            pos1_vocab.vectors[:len(cmp_vocab.keys), :],
            threshold=0.25)

        nengo.Connection(net.frm_compare, net.am_compare.input,
                         synapse=None)
        nengo.Connection(net.am_compare.output, net.compare_out,
                         synapse=None)

        # ----------------------- Inputs and Outputs --------------------------
        net.pos1_to_pos = net.am_p1.output
        net.pos_to_pos1 = net.am_p2.output
        net.pos1_to_num = net.am_n1.output
        net.num_to_pos1 = net.am_n2.output

    return net
