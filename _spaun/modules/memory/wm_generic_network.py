import numpy as np
import nengo

from ...configurator import cfg


def WM_Generic_Network(vocab, sp_add_matrix, net=None, net_label="MB"):
    if net is None:
        net = nengo.Network(label=net_label)

    if sp_add_matrix is None:
        sp_add_matrix = np.eye(vocab.dimensions)

    with net:
        # Memory block (MBA - long term memory (rehearsal),
        #               MBB - short term memory (decay))
        sel_in = cfg.make_selector(3, default_sel=0,
                                   label=net_label + 'Input Selector')

        mb_gate = nengo.Node(size_in=1, label=net_label + ' Gate Node')
        mb_reset = nengo.Node(size_in=1, label=net_label + ' Reset Node')

        mba = cfg.make_mem_block(vocab=vocab, reset_key=0,
                                 label=net_label + 'A (Rehearsal)',
                                 make_ens_func=cfg.make_ens_array)
        mbb = cfg.make_mem_block(vocab=vocab, fdbk_transform=cfg.mb_decay_val,
                                 reset_key=0, label=net_label + 'B (Decay)',
                                 make_ens_func=cfg.make_ens_array)

        nengo.Connection(sel_in.output, mba.input,
                         transform=cfg.mb_rehearsalbuf_input_scale)
        nengo.Connection(sel_in.output, mbb.input,
                         transform=cfg.mb_decaybuf_input_scale)

        # Feedback gating ensembles. NOTE: Needs thresholded input as gate
        mba_fdbk_gate = \
            cfg.make_spa_ens_array_gate(label=net_label + 'A Fdbk Gate')
        mbb_fdbk_gate = \
            cfg.make_spa_ens_array_gate(label=net_label + 'B Fdbk Gate')

        mb_fdbk_gate = nengo.Node(size_in=1, label=net_label + ' Fdbk Gate')
        nengo.Connection(mb_fdbk_gate, mba_fdbk_gate.gate, synapse=None)
        nengo.Connection(mb_fdbk_gate, mbb_fdbk_gate.gate, synapse=None)

        nengo.Connection(mba.output, mba_fdbk_gate.input,
                         transform=cfg.mb_fdbk_val)
        nengo.Connection(mba_fdbk_gate.output, mba.input)
        nengo.Connection(mbb.output, mbb_fdbk_gate.input)
        nengo.Connection(mbb_fdbk_gate.output, mbb.input)

        nengo.Connection(mb_gate, mba.gate, synapse=None)
        nengo.Connection(mb_gate, mbb.gate, synapse=None)

        nengo.Connection(mb_reset, mba.reset, synapse=None)
        nengo.Connection(mb_reset, mbb.reset, synapse=None)

        # ---------- Network input and outputs ----------
        net.input = sel_in.input0
        net.side_load = sel_in.input2

        net.sel0 = sel_in.sel0
        net.sel1 = sel_in.sel1
        net.sel2 = sel_in.sel2

        net.mb_reh = mba.output
        net.mb_dcy = mbb.output

        net.output = nengo.Node(size_in=vocab.dimensions,
                                label=net_label + ' Out Node')
        nengo.Connection(mba.output, net.output, synapse=None)
        nengo.Connection(mbb.output, net.output, synapse=None)

        nengo.Connection(net.output, sel_in.input1, transform=sp_add_matrix)

        net.gate = mb_gate
        net.reset = mb_reset
        net.fdbk_gate = mb_fdbk_gate

        # ###### DEBUG #######
        net.mba = mba
        net.mbb = mbb

    return net
