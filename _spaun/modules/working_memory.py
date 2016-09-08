from warnings import warn

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from .._networks import Selector
from .memory import WM_Generic_Network, WM_Averaging_Network


class WorkingMemory(Module):
    def __init__(self, label="Working Memory", seed=None,
                 add_to_container=None):
        super(WorkingMemory, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        # Memory input selector
        self.select_in = cfg.make_selector(
            2, default_sel=0, make_ens_func=cfg.make_ens_array)

        # Gate signal selector
        self.select_gate = Selector(cfg.n_neurons_ens, dimensions=1,
                                    num_items=2, make_ens_func=nengo.Ensemble,
                                    gate_gain=10, default_sel=0,
                                    threshold_sel_in=True)

        # 'ADD' transform for WM
        sp_add_matrix = (vocab.add_sp.get_convolution_matrix() *
                         (0.25 / cfg.mb_rehearsalbuf_input_scale +
                          0.25 / (cfg.mb_decaybuf_input_scale - 0.15)))

        self.num0_bias_node = nengo.Node(vocab.main.parse('POS1*ZER').v,
                                         label="POS1*ZER")

        self.gate_sig_bias = cfg.make_thresh_ens_net(label='Gate Sig Bias')
        # Bias the -1.5 neg_atn during decoding phase (when there is no input)
        # nengo.Connection(self.gate_sig_bias.output, self.select_gate.input0)

        self.cnt_gate_sig = cfg.make_thresh_ens_net(0.5, label='Cnt Gate Sig')
        nengo.Connection(self.cnt_gate_sig.output, self.select_gate.input0,
                         transform=cfg.mb_gate_scale)

        # Memory block 1
        self.mb1_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB1")
        nengo.Connection(self.select_in.output, self.mb1_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb1_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb1_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb1_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb1_net.gate,
        #                  transform=1.0)

        self.mb1 = self.mb1_net.output

        # Memory block 2
        self.mb2_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB2")
        nengo.Connection(self.select_in.output, self.mb2_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb2_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb2_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb2_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb2_net.gate,
        #                  transform=1.0)

        self.mb2 = self.mb2_net.output

        # Memory block 3
        self.mb3_net = WM_Generic_Network(vocab.main, sp_add_matrix,
                                          net_label="MB3")
        nengo.Connection(self.select_in.output, self.mb3_net.input,
                         synapse=None)
        nengo.Connection(self.select_gate.output, self.mb3_net.gate)
        nengo.Connection(self.num0_bias_node, self.mb3_net.side_load,
                         synapse=None)
        nengo.Connection(self.gate_sig_bias.output, self.mb3_net.gate,
                         transform=cfg.mb_neg_attn_scale * 0.75, synapse=0.01)
        # nengo.Connection(self.cnt_gate_sig.output, self.mb3_net.gate,
        #                  transform=1.0)

        self.mb3 = self.mb3_net.output

        # Memory block Ave (MBAve)
        self.mbave_net = WM_Averaging_Network(vocab.main)
        self.mbave = self.mbave_net.output

        # Define network inputs and outputs
        self.input = self.select_in.input0
        self.data_input = self.select_in.input1

        self.gate_in = self.select_gate.input0
        self.data_gate_in = self.select_gate.input1

        # ----- Set up module vocab inputs and outputs -----
        self.outputs = dict(mb1=(self.mb1, vocab.enum),
                            mb2=(self.mb2, vocab.enum),
                            mb3=(self.mb3, vocab.enum),
                            mbave=(self.mbave, vocab.enum))

        # ### DEBUG ###
        self.gate_in_neg_att_dbg = nengo.Node(size_in=1)
        self.gate_in_vis_dbg = nengo.Node(size_in=1)
        self.gate_in_2 = nengo.Node(size_in=1)
        nengo.Connection(self.gate_sig_bias.output, self.gate_in_2)
        nengo.Connection(self.cnt_gate_sig.output, self.gate_in_2,
                         transform=cfg.mb_gate_scale)
        self.gate_sel_none = self.select_gate.sel_none
        self.gate_sel_node0 = self.select_gate.sel_nodes[0]
        self.gate_sel_node1 = self.select_gate.sel_nodes[1]
        self.gate_sel_node1_in = nengo.Node(size_in=1)
        # ### DEBUG ###

    def setup_connections(self, parent_net):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            item_mb_gate_sp_vecs = \
                vocab.main.parse('+'.join(vocab.num_sp_strs)).v
            item_mb_rst_sp_vecs = vocab.main.parse('A+OPEN').v

            # ###### Common Gate Input Signal ######
            nengo.Connection(p_net.vis.output, self.gate_in,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_gate_sp_vecs])

            # ### DEBUG ###
            nengo.Connection(p_net.vis.output, self.gate_in_vis_dbg,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_gate_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention,
                             self.gate_in_neg_att_dbg, transform=-2.0,
                             synapse=0.01)
            nengo.Connection(p_net.vis.output, self.gate_in_2,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_gate_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention,
                             self.gate_in_2,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)
            # ### DEBUG ###

            # ###### MB1 ########
            nengo.Connection(p_net.vis.output, self.mb1_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention, self.mb1_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MB2 ########
            nengo.Connection(p_net.vis.output, self.mb2_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention, self.mb2_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MB3 ########
            nengo.Connection(p_net.vis.output, self.mb3_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        item_mb_rst_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention, self.mb3_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            # ###### MBAve ########
            ave_mb_gate_sp_vecs = vocab.main.parse('CLOSE').v
            ave_mb_rst_sp_vecs = vocab.main.parse('A').v

            nengo.Connection(p_net.vis.output, self.mbave_net.gate,
                             transform=[cfg.mb_gate_scale *
                                        ave_mb_gate_sp_vecs])
            nengo.Connection(p_net.vis.neg_attention,
                             self.mbave_net.gate,
                             transform=-cfg.mb_neg_attn_scale, synapse=0.01)

            nengo.Connection(p_net.vis.output, self.mbave_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        ave_mb_rst_sp_vecs])
        else:
            warn("WorkingMemory Module - Cannot connect from 'vis'")

        # Set up connections from production system module
        if hasattr(p_net, 'ps'):
            # ###### INPUT SELECTOR #######
            instr_task_sp_vecs = vocab.main.parse('INSTR').v
            nengo.Connection(p_net.ps.task, self.select_in.sel1,
                             transform=[instr_task_sp_vecs])
            nengo.Connection(p_net.ps.task, self.select_gate.sel1,
                             transform=[instr_task_sp_vecs])

            # ### DEBUG ###
            nengo.Connection(p_net.ps.task, self.gate_sel_node1_in,
                             transform=[instr_task_sp_vecs])
            # ### DEBUG ###

            # ###### MB1 ########
            mb1_no_gate_sp_vecs = \
                vocab.main.parse('X+QAP+QAK+TRANS1+TRANS2+CNT0+L').v
            nengo.Connection(p_net.ps.state, self.mb1_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb1_no_gate_sp_vecs])
            nengo.Connection(p_net.ps.task, self.mb1_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb1_no_gate_sp_vecs])

            mb1_no_reset_sp_vecs = \
                vocab.main.parse('QAP+QAK+TRANS1+CNT0+CNT1').v
            nengo.Connection(p_net.ps.state, self.mb1_net.reset,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb1_no_reset_sp_vecs])

            mb1_sel_1_sp_vecs = vocab.main.parse('CNT1').v
                # Use *ONE connection in the CNT1 state  # noqa
            nengo.Connection(p_net.ps.state, self.mb1_net.sel1,
                             transform=[mb1_sel_1_sp_vecs])
            nengo.Connection(p_net.ps.state, self.mb1_net.fdbk_gate,
                             transform=[mb1_sel_1_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(p_net.ps.task, self.mb1_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MB2 ########
            mb2_no_gate_sp_vecs = \
                vocab.main.parse('X+TRANS0+TRANS2+CNT1+L').v
            nengo.Connection(p_net.ps.state, self.mb2_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb2_no_gate_sp_vecs])
            nengo.Connection(p_net.ps.task, self.mb2_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb2_no_gate_sp_vecs])

            mb2_no_reset_sp_vecs = \
                vocab.main.parse('QAP+QAK+TRANS1+TRANS2+CNT1').v
            nengo.Connection(p_net.ps.state, self.mb2_net.reset,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb2_no_reset_sp_vecs])

            mb2_sel_1_sp_vecs = vocab.main.parse('0').v
                # TODO: Make configurable? Use *ONE connection in the none # noqa
            nengo.Connection(p_net.ps.state, self.mb2_net.sel1,
                             transform=[mb2_sel_1_sp_vecs])
            nengo.Connection(p_net.ps.state, self.mb2_net.fdbk_gate,
                             transform=[mb2_sel_1_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(p_net.ps.task, self.mb2_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MB3 ########
            mb3_no_gate_sp_vecs = \
                vocab.main.parse('X+QAP+QAK+TRANS0+TRANS1+L').v
            nengo.Connection(p_net.ps.state, self.mb3_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb3_no_gate_sp_vecs])
            nengo.Connection(p_net.ps.task, self.mb3_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb3_no_gate_sp_vecs])

            mb3_no_reset_sp_vecs = vocab.main.parse('CNT1').v
            nengo.Connection(p_net.ps.state, self.mb3_net.reset,
                             transform=[cfg.mb_neg_gate_scale *
                                        mb3_no_reset_sp_vecs])

            mb3_sel_1_sp_vecs = vocab.main.parse('CNT1').v
                # Use *ONE connection in the CNT1 state  # noqa
            nengo.Connection(p_net.ps.state, self.mb3_net.sel1,
                             transform=[mb3_sel_1_sp_vecs])
            nengo.Connection(p_net.ps.state, self.mb3_net.fdbk_gate,
                             transform=[mb3_sel_1_sp_vecs])

            mb3_sel_2_sp_vecs = vocab.main.parse('CNT0').v
                # Use POS1*ONE connection for CNT0 state  # noqa
            nengo.Connection(p_net.ps.state, self.mb3_net.sel2,
                             transform=[mb3_sel_2_sp_vecs])

            # Disable memory feedback connection for INSTR task
            nengo.Connection(p_net.ps.task, self.mb3_net.fdbk_gate,
                             transform=[instr_task_sp_vecs])

            # ###### MBAVe ########
            mbave_no_gate_sp_vecs = \
                vocab.main.parse('X+QAP+QAK+TRANS0+L').v
            nengo.Connection(p_net.ps.state, self.mbave_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mbave_no_gate_sp_vecs])
            nengo.Connection(p_net.ps.task, self.mbave_net.gate,
                             transform=[cfg.mb_neg_gate_scale *
                                        mbave_no_gate_sp_vecs])

            mbave_do_reset_sp_vecs = vocab.main.parse('X').v
            nengo.Connection(p_net.ps.task, self.mbave_net.reset,
                             transform=[cfg.mb_gate_scale *
                                        mbave_do_reset_sp_vecs])

            # ###### Gate Signal Bias ######
            gate_sig_bias_enable_sp_vecs = vocab.main.parse('CNT').v
                # Only enable gate signal bias for dec=CNT  # noqa
            nengo.Connection(p_net.ps.dec, self.gate_sig_bias.input,
                             transform=[gate_sig_bias_enable_sp_vecs],
                             synapse=0.01)
        else:
            warn("WorkingMemory Module - Cannot connect from 'ps'")

        # Set up connections from encoding module
        if hasattr(p_net, 'enc'):
            nengo.Connection(p_net.enc.enc_output, self.input)
        else:
            warn("WorkingMemory Module - Cannot connect from 'enc'")

        # Set up connections from transformation system module
        if hasattr(p_net, 'trfm'):
            nengo.Connection(p_net.trfm.output, self.mbave_net.input)
        else:
            warn("WorkingMemory Module - Cannot connect from 'trfm'")

        # Set up connections from instruction processing module
        if hasattr(p_net, 'instr'):
            nengo.Connection(p_net.instr.output, self.data_input)
            nengo.Connection(p_net.instr.data_gate_sig, self.data_gate_in,
                             transform=2.0, synapse=0.01)
            # Note: Need to filter the data_gate_sig signal a bit. Change
            #       detect mechanism is a little sensitive.
        else:
            warn("WorkingMemory Module - Cannot connect from 'instr'")

        # Set up connections from motor module (for counting task)
        if hasattr(parent_net, 'mtr'):
            # nengo.Connection(parent_net.mtr.ramp_50_75,
            #                  self.cnt_gate_sig.input, transform=2.0,
            #                  synapse=0.01)
            nengo.Connection(parent_net.mtr.ramp_reset_hold,
                             self.cnt_gate_sig.input, transform=2.0,
                             synapse=0.01)
        else:
            warn("WorkingMemory Module - Could not connect from 'mtr'")


class WorkingMemoryDummy(WorkingMemory):
    def __init__(self):
        super(WorkingMemoryDummy, self).__init__()
        self.init_module()

    @with_self
    def init_module(self):
        # Memory input node
        self.mem_in = nengo.Node(size_in=vocab.sp_dim,
                                 label='WM Module In Node')

        self.gate_sig_bias = nengo.Node(size_in=1)

        # Memory block 1 (MB1A - long term memory, MB1B - short term memory)
        self.mb1 = \
            nengo.Node(output=vocab.main.parse('POS1*FOR+POS2*THR+POS3*FOR').v)
        self.mb1_net.gate = nengo.Node(size_in=1, label='MB1 Gate Node')
        self.mb1_net.reset = nengo.Node(size_in=1, label='MB1 Reset Node')

        self.sel_mb1_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=vocab.sp_dim,
                                            n_neurons=vocab.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb1_in.input0, synapse=None)

        # Memory block 2 (MB2A - long term memory, MB2B - short term memory)
        self.mb2 = nengo.Node(output=vocab.main.parse('POS1*THR').v)
        self.mb2_gate = nengo.Node(size_in=1, label='MB2 Gate Node')
        self.mb2_reset = nengo.Node(size_in=1, label='MB2 Reset Node')

        self.sel_mb2_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=vocab.sp_dim,
                                            n_neurons=vocab.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb2_in.input0, synapse=None)

        # Memory block 3 (MB3A - long term memory, MB3B - short term memory)
        self.mb3 = nengo.Node(output=vocab.main.parse('POS1*ONE').v)
        self.mb3_gate = nengo.Node(size_in=1, label='MB3 Gate Node')
        self.mb3_reset = nengo.Node(size_in=1, label='MB3 Reset Node')

        self.sel_mb3_in = cfg.make_selector(3, default_sel=0, n_ensembles=1,
                                            ens_dimensions=vocab.sp_dim,
                                            n_neurons=vocab.sp_dim)
        nengo.Connection(self.mem_in, self.sel_mb3_in.input0, synapse=None)

        # Memory block Ave (MBAve)
        self.mbave_in = nengo.Node(size_in=vocab.sp_dim)
        self.mbave = nengo.Node(output=vocab.main.parse('~POS1').v)
        self.mbave_gate = nengo.Node(size_in=1)
        self.mbave_reset = nengo.Node(size_in=1)

        # Define network inputs and outputs
        # ## TODO: Fix this! (update to include selector and what not)
        self.input = self.mem_in
