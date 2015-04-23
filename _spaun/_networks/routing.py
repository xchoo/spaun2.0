import nengo


def make_route_connections_common(net, ens_class, num_items, gate_gain,
                                  default_sel=None, **ens_args):
    with net:
        bias_node = nengo.Node(output=1)

        net.sel_none = nengo.Ensemble(20, 1)
        nengo.Connection(bias_node, net.sel_none, synapse=None)

        for n in range(num_items):
            sel_node = nengo.Node(size_in=1)
            ens = ens_class(**ens_args)

            nengo.Connection(sel_node, net.sel_none.neurons,
                             transform=([[-gate_gain]] *
                                        net.sel_none.n_neurons))

            if isinstance(ens, nengo.Network):
                if n != default_sel:
                    for e in ens.all_ensembles:
                        nengo.Connection(
                            net.sel_none, e.neurons,
                            transform=[[-gate_gain]] * e.n_neurons)
                for sn in net.sel_nodes:
                    for e in ens.all_ensembles:
                        nengo.Connection(
                            sn, e.neurons,
                            transform=[[-gate_gain]] * e.n_neurons)
                for ee in net.ens_elements:
                    for e in ee.all_ensembles:
                        nengo.Connection(
                            sel_node, e.neurons,
                            transform=[[-gate_gain]] * e.n_neurons)
            else:
                if n != default_sel:
                    nengo.Connection(net.sel_none, ens.neurons,
                                     transform=[[-gate_gain]] * ens.n_neurons)
                for sn in net.sel_nodes:
                    nengo.Connection(
                        sn, ens.neurons,
                        transform=[[-gate_gain]] * ens.n_neurons)
                for ee in net.ens_elements:
                    nengo.Connection(
                        sel_node, ee.neurons,
                        transform=[[-gate_gain]] * ee.n_neurons)

            net.ens_elements.append(ens)
            net.sel_nodes.append(sel_node)

            setattr(net, 'sel%i' % n, sel_node)
            setattr(net, 'ens%i' % n, ens)


class Selector(nengo.Network):
    def __init__(self, ens_class, num_items, dimensions, gate_gain=3,
                 default_sel=None,
                 label=None, seed=None, add_to_container=None, **ens_args):

        super(Selector, self).__init__(label, seed, add_to_container)

        self.ens_elements = []
        self.sel_nodes = []

        self.dimensions = dimensions

        make_route_connections_common(self, ens_class, num_items, gate_gain,
                                      default_sel=default_sel, **ens_args)

        with self:
            self.output = nengo.Node(size_in=self.dimensions)

            for n, ens in enumerate(self.ens_elements):
                if isinstance(ens, nengo.Network):
                    nengo.Connection(ens.output, self.output, synapse=None)
                    setattr(self, 'input%i' % n, ens.input)
                else:
                    nengo.Connection(ens, self.output, synapse=None)
                    setattr(self, 'input%i' % n, ens)


class Router(nengo.Network):
    def __init__(self, ens_class, num_items, dimensions, gate_gain=3,
                 default_sel=None,
                 label=None, seed=None, add_to_container=None, **ens_args):

        super(Router, self).__init__(label, seed, add_to_container)

        self.ens_elements = []
        self.sel_nodes = []

        self.dimensions = dimensions

        make_route_connections_common(self, ens_class, num_items, gate_gain,
                                      default_sel=default_sel, **ens_args)

        with self:
            self.input = nengo.Node(size_in=self.dimensions)

            for n, ens in enumerate(self.ens_elements):
                if isinstance(ens, nengo.Network):
                    nengo.Connection(self.input, ens.input, synapse=None)
                    setattr(self, 'output%i' % n, ens.output)
                else:
                    nengo.Connection(self.input, ens, synapse=None)
                    setattr(self, 'output%i' % n, ens)
