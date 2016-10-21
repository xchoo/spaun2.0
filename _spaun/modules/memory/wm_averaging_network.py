import numpy as np
import nengo

from ...configurator import cfg


def WM_Averaging_Network(vocab, net=None, net_label="MBAve"):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        mb = cfg.make_mem_block(vocab=vocab, label=net_label, reset_key=0,
                                represent_identity=True, identity_radius=2.0)

        # Feedback from mb ave to mb ave = 1 - alpha
        nengo.Connection(mb.output, mb.input,
                         transform=(1 - cfg.trans_ave_scale))

        # Initial input to mb ave = input * (1 - alpha)
        # - So that mb ave is initialized with full input when empty
        mb_in_init = cfg.make_spa_ens_array_gate()
        nengo.Connection(mb_in_init.output, mb.input,
                         transform=(1 - cfg.trans_ave_scale))

        # Output norm calculation for mb ave (to shut off init input to mbave)
        mb.mem2.mem.add_output('squared', lambda x: x * x)
        mb_norm = cfg.make_thresh_ens_net()
        nengo.Connection(mb.mem2.mem.squared, mb_norm.input,
                         transform=np.ones((1, vocab.dimensions)))
        nengo.Connection(mb_norm.output, mb_in_init.gate)

        # ---------- Network input and outputs ----------
        net.input = nengo.Node(size_in=vocab.dimensions,
                               label=net_label + 'In Node')

        nengo.Connection(net.input, mb.input, synapse=None,
                         transform=cfg.trans_ave_scale)  # mb = alpha * input
        nengo.Connection(net.input, mb_in_init.input, synapse=None)

        net.output = mb.output
        net.gate = mb.gate
        net.reset = mb.reset

    return net
