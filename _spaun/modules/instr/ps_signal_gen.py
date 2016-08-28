import numpy as np
import nengo
from nengo.dists import Choice, Uniform

from ..._networks import DetectChange
from ...configurator import cfg


def PS_Signal_Generator_Network(vocab, net_label, cleanup_threshold=0.3):
    # Returns a network that generates the appropriate values and signals for
    # the production system. Takes in the sp output from the instruction data
    # decoding network and produces a cleaned version (using `vocab`), and
    # gating signals to drive the appropriate prod system memories.
    net = nengo.Network(label=net_label)

    with net:
        net.input = nengo.Node(size_in=vocab.dimensions)

        # ----- AM Cleanup for input vector -----
        am = cfg.make_assoc_mem(vocab.vectors, threshold=cleanup_threshold,
                                inhibitable=True)
        nengo.Connection(net.input, am.input)

        # ----- Zero SP detection network -----
        # Detect zero (vector with zeros for all elements) vector to disable
        # gate signal when input vector is zero vector.
        bias_node = nengo.Node(1)
        zero_detect = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                     encoders=Choice([[1]]),
                                     intercepts=Uniform(0.5, 1))
        nengo.Connection(bias_node, zero_detect)
        nengo.Connection(net.input, zero_detect,
                         transform=-np.sum(vocab.vectors, axis=0)[:, None].T)

        # ----- Gate signal generation network -----
        # Generate a gating signal (high) when a change in the input vector
        # is detected. Note: no gating signal is generated if input vector
        # changes to zero vector.
        gate_sig_gen = DetectChange(
            dimensions=vocab.dimensions, blank_output_value=0, diff_scale=1.0,
            item_magnitude=cfg.get_optimal_sp_radius(vocab.dimensions))
        nengo.Connection(am.output, gate_sig_gen.input)
        nengo.Connection(zero_detect, gate_sig_gen.change_detect, transform=-5)
        nengo.Connection(am.inhibit, gate_sig_gen.change_detect, transform=-5)

        # ----- Network outputs -----
        net.output = am.output
        net.inhibit = am.inhibit
        net.gate_sig = gate_sig_gen.output
        net.gate_sig_in = gate_sig_gen.change_detect

    return net
