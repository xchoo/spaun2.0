import numpy as np

import nengo

from ...._networks import AssociativeMemory as AM


def LIFImagenetClassifier(vis_data, concept_sps, net=None):
    if net is None:
        net = nengo.Network(label="LIF Imagenet Classifier")

    with net:
        # --- Common inputs and outputs ---
        net.input = nengo.Node(size_in=vis_data.output_dimensions)
        net.inhibit = nengo.Node(size_in=1)
        net.output = nengo.Node(size_in=concept_sps.shape[1])
        net.output_utilities = \
            nengo.Node(size_in=(vis_data.spaun_sym_num_classes +
                                vis_data.imagenet_num_classes))

        # --- Spaun symbols AM network ---
        spaun_sym_am_net = AM(vis_data.sps_spaun_sym,
                              concept_sps[:vis_data.spaun_sym_num_classes],
                              threshold=vis_data.spaun_am_threshold,
                              n_neurons=50, inhibitable=True)
        spaun_sym_am_net.add_wta_network(3.5)

        # Shift and rescale input vectors (remap utilities) for better
        # input classification (0 -> 1 to 0.9 -> 1)
        spaun_sym_am_net.add_input_mapping(
            'bias', np.eye(vis_data.spaun_sym_num_classes))
        spaun_sym_am_input_bias = \
            nengo.Node(np.ones(vis_data.spaun_sym_num_classes))
        nengo.Connection(spaun_sym_am_input_bias, spaun_sym_am_net.bias,
                         transform=vis_data.spaun_am_input_bias)

        nengo.Connection(net.input[:vis_data.spaun_sym_out_dimensions],
                         spaun_sym_am_net.input,
                         transform=vis_data.spaun_am_input_scale, synapse=0.01)

        # Misc inputs and outputs
        nengo.Connection(net.inhibit, spaun_sym_am_net.inhibit, synapse=None)
        nengo.Connection(spaun_sym_am_net.output, net.output,
                         synapse=None)
        nengo.Connection(spaun_sym_am_net.output_utilities,
                         net.output_utilities[:vis_data.spaun_sym_num_classes])

        # --- Imagenet AM network ---
        imagenet_am_net = AM(vis_data.sps_fc1000,
                             concept_sps[vis_data.spaun_sym_num_classes:],
                             threshold=vis_data.imagenet_am_threshold,
                             n_neurons=50, inhibitable=True)
        imagenet_am_net.add_cleanup_output(replace_output=True,
                                           inhibit_scale=10.0)

        nengo.Connection(net.input[vis_data.spaun_sym_out_dimensions:],
                         imagenet_am_net.input,
                         transform=1.0 / vis_data.imagenet_sps_means,
                         synapse=0.01)
        nengo.Connection(net.inhibit, imagenet_am_net.inhibit, synapse=None)
        nengo.Connection(imagenet_am_net.output, net.output,
                         synapse=None)
        nengo.Connection(imagenet_am_net.output_utilities,
                         net.output_utilities[vis_data.spaun_sym_num_classes:])

        # Inhibit imagenet AM network when Spaun symbol AM network has
        # succesfully classified an output
        nengo.Connection(
            spaun_sym_am_net.output_utilities, imagenet_am_net.inhibit,
            transform=5 * np.ones((1, vis_data.spaun_sym_num_classes)))
    return net
