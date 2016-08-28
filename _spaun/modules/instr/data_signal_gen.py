import numpy as np
import nengo
from nengo.dists import Choice, Uniform

from ..._networks import DetectChange
from ...configurator import cfg


def Data_Signal_Generator_Network(vocab, net_label):
    # Returns a network that generates the appropriate values and signals for
    # the production system. Takes in the sp output from the instruction data
    # decoding network and produces a cleaned version (using `vocab`), and
    # gating signals to drive the appropriate prod system memories.
    net = nengo.Network(label=net_label)

    with net:
        net.input = nengo.Node(size_in=vocab.dimensions)

        # ----- Ensemble Array for calculating vector norm -----
        ea = cfg.make_ens_array()
        ea.add_output('sqr', lambda x: x ** 2)
        nengo.Connection(net.input, ea.input)

        # ----- Zero SP detection network -----
        # Detect zero (vector with zeros for all elements) vector to disable
        # gate signal when input vector is zero vector.
        bias_node = nengo.Node(1)
        zero_detect = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                     encoders=Choice([[1]]),
                                     intercepts=Uniform(0.5, 1))
        nengo.Connection(bias_node, zero_detect)
        nengo.Connection(ea.sqr, zero_detect,
                         transform=[[-1] * vocab.dimensions])

        # ----- Gate signal generation network -----
        # Generate a gating signal (high) when a change in the input vector
        # is detected. Note: no gating signal is generated if input vector
        # changes to zero vector.
        gate_sig_gen = DetectChange(
            dimensions=vocab.dimensions, blank_output_value=0, diff_scale=1.0,
            item_magnitude=cfg.get_optimal_sp_radius(vocab.dimensions))
        nengo.Connection(ea.output, gate_sig_gen.input)
        nengo.Connection(zero_detect, gate_sig_gen.change_detect,
                         transform=-10)

        # ----- Network outputs -----
        net.output = ea.output
        net.gate_sig = gate_sig_gen.output
        net.gate_sig_in = gate_sig_gen.change_detect

    return net
