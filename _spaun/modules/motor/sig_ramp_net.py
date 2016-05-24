import nengo

from ...configurator import cfg


def Ramp_Signal_Network(net=None, net_label='RAMP SIGNAL'):
    if net is None:
        net = nengo.Network(label=net_label)

    with net:
        bias_node = nengo.Node(output=1)

        # Ramp init hold
        ramp_init_hold = \
            cfg.make_thresh_ens_net(0.07, thresh_func=lambda x: x)
        nengo.Connection(ramp_init_hold.output,
                         ramp_init_hold.input)
        nengo.Connection(bias_node, ramp_init_hold.input,
                         transform=-cfg.mtr_ramp_reset_hold_transform)

        # Ramp reset hold
        ramp_reset_hold = \
            cfg.make_thresh_ens_net(0.07, thresh_func=lambda x: x)
        nengo.Connection(ramp_reset_hold.output,
                         ramp_reset_hold.input)
        nengo.Connection(bias_node, ramp_reset_hold.input,
                         transform=-cfg.mtr_ramp_reset_hold_transform)

        # Ramp integrator go signal (stops ramp integrator when <= 0.5)
        ramp_int_go = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, ramp_int_go.input)
        nengo.Connection(ramp_init_hold.output, ramp_int_go.input,
                         transform=-2)
        nengo.Connection(ramp_reset_hold.output, ramp_int_go.input,
                         transform=-2)

        # Ramp integrator stop signal (inverse of ramp integrator go signal)
        ramp_int_stop = cfg.make_thresh_ens_net()
        nengo.Connection(bias_node, ramp_int_stop.input)
        nengo.Connection(ramp_int_go.output, ramp_int_stop.input, transform=-2)

        # Ramp integrator
        ramp_integrator = nengo.Ensemble(cfg.n_neurons_cconv, 1, radius=1.1)

        # -- Note: We could use the ramp_int_go signal here, but it is noisier
        #    than the motor_go signal
        nengo.Connection(ramp_integrator, ramp_integrator,
                         synapse=cfg.mtr_ramp_synapse)

        # Ramp integrator reset circuitry -- Works like the difference gate in
        # the gated integrator
        ramp_int_reset = nengo.Ensemble(cfg.n_neurons_ens, 1)
        nengo.Connection(ramp_integrator, ramp_int_reset)
        nengo.Connection(ramp_int_reset, ramp_integrator,
                         transform=-10, synapse=cfg.mtr_ramp_synapse)
        nengo.Connection(ramp_int_go.output, ramp_int_reset.neurons,
                         transform=[[-3]] * cfg.n_neurons_ens)

        # Ramp end reset signal generator -- Signals when the ramp integrator
        # reaches the top of the ramp slope.
        ramp_reset_thresh = cfg.make_thresh_ens_net(0.91, radius=1.1)
        nengo.Connection(ramp_reset_thresh.output, ramp_reset_hold.input,
                         transform=2.5, synapse=0.015)

        # Misc ramp threshold outputs
        ramp_75 = cfg.make_thresh_ens_net(0.75)
        ramp_50_75 = cfg.make_thresh_ens_net(0.5)

        nengo.Connection(ramp_integrator, ramp_reset_thresh.input)
        nengo.Connection(ramp_integrator, ramp_75.input)
        nengo.Connection(ramp_integrator, ramp_50_75.input)
        nengo.Connection(ramp_75.output, ramp_50_75.input, transform=-3)

        # ----------------------- Inputs and Outputs --------------------------
        net.ramp = ramp_integrator
        net.ramp_50_75 = ramp_50_75.output

        net.go = ramp_int_go.input
        net.end = ramp_reset_thresh.input
        net.init = ramp_init_hold.input
        net.reset = ramp_reset_hold.input

        net.stop = ramp_int_stop.output
        net.init_hold = ramp_init_hold.output
        net.reset_hold = ramp_reset_hold.output

    return net
