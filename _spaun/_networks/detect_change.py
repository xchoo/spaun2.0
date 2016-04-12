import nengo
from nengo.dists import Choice, Uniform


def DetectChange(net=None, dimensions=1, n_neurons=50, diff_scale=0.2,
                 item_magnitude=1):
    # item_magnitude: expected magnitude of one pixel in the input image

    if net is None:
        net = nengo.Network(label="Detect Change Network")

    with net:
        net.input = nengo.Node(size_in=dimensions)

        # Negative attention signal generation. Generates a high valued signal
        # when the input is changing or when there is nothing being presented
        # to the visual system
        input_diff = nengo.networks.EnsembleArray(n_neurons, dimensions,
                                                  label='Input Differentiator',
                                                  intercepts=Uniform(0.1, 1))
        input_diff.add_output('abs', lambda x: abs(x))
        nengo.Connection(net.input, input_diff.input, synapse=0.005,
                         transform=1.0 / item_magnitude)
        nengo.Connection(net.input, input_diff.input, synapse=0.020,
                         transform=-1.0 / item_magnitude)

        #######################################################################
        net.output = nengo.Ensemble(n_neurons, 1,
                                    intercepts=Uniform(0.5, 1),
                                    encoders=Choice([[1]]),
                                    eval_points=Uniform(0.5, 1))
        nengo.Connection(input_diff.abs, net.output, synapse=0.005,
                         transform=[[diff_scale] * dimensions])

        item_detect = nengo.Ensemble(n_neurons * 3, 1)
        nengo.Connection(net.input, item_detect, synapse=0.005,
                         transform=[[1.0 / item_magnitude] * dimensions])
        nengo.Connection(item_detect, net.output, synapse=0.005,
                         function=lambda x: 1 - abs(x))

        blank_detect = nengo.Ensemble(n_neurons, 1,
                                      intercepts=Uniform(0.7, 1),
                                      encoders=Choice([[1]]),
                                      eval_points=Uniform(0.7, 1))
        nengo.Connection(item_detect, blank_detect, synapse=0.005,
                         function=lambda x: 1 - abs(x))
        #######################################################################

        # Delay ensemble needed to smooth out transition from blank to
        # change detection
        blank_detect_delay = nengo.Ensemble(n_neurons, 1,
                                            intercepts=Uniform(0.5, 1),
                                            encoders=Choice([[1]]),
                                            eval_points=Uniform(0.5, 1),
                                            label='Blank Detect')
        nengo.Connection(blank_detect, blank_detect_delay, synapse=0.03)
        nengo.Connection(blank_detect, net.output, synapse=0.01,
                         transform=2)

        # ### DEBUG ####
        net.input_diff = input_diff.output
        net.item_detect = item_detect
        net.blank_detect = blank_detect

    return net
