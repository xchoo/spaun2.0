import nengo

from ...._networks import AssociativeMemory as AM


def LIFConvNetClassifier(vis_data, concept_sps, net=None):
    if net is None:
        net = nengo.Network(label="LIF Vision Classifier")

    with net:
        am_net = AM(vis_data.sps_fc10, concept_sps,
                    threshold=vis_data.am_threshold,
                    n_neurons=50, inhibitable=True)

        am_net.add_wta_network(3.5)
        am_net.add_cleanup_output(replace_output=True)

        # --- Set up input and outputs to the LIF vision system
        net.input = am_net.input
        net.inhibit = am_net.inhibit
        net.output = am_net.cleaned_output
        net.output_utilities = am_net.cleaned_output_utilities

    return net
