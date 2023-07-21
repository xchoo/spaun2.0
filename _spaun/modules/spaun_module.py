from warnings import warn

import nengo
from nengo import Network
from nengo.spa.module import Module
from nengo.utils.numpy import is_integer
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from ..sockets import UDPSendReceiveSocket


class SpaunModule(Module):
    def __init__(self, id_str, ind_num, label="", seed=None, add_to_container=None):
        super(SpaunModule, self).__init__(label, seed, add_to_container)

        # String identifier for this module. Used to set attribute name in parent
        # network.
        self.id_str = id_str

        # Dimension to input / output mapping
        self.dim_map = {}

        # Dictionary to keep track of external module connections
        self.module_conns = {}

        # Port numbers for multi-process implementations
        base_port = cfg.seed % 65535
        if base_port < 2048:
            base_port += 2048
        self.mp_hub_port = base_port + ind_num * 2
        self.mp_node_port = base_port + ind_num * 2 + 1

        # ## DEBUG ##
        # print(">>", id_str, base_port, self.mp_hub_port, self.mp_node_port)

        self.init_module()
        self.setup_inputs_and_outputs()
        self.setup_spa_inputs_and_outputs()

    @with_self
    def init_module(self):
        self.config[nengo.Ensemble].max_rates = cfg.max_rates
        self.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        self.config[nengo.Connection].synapse = cfg.pstc

    def expose_input(self, name, obj):
        if is_integer(obj):
            with self:
                obj = nengo.Node(size_in=obj)

        self.dim_map[f"inp_{name}"] = obj.size_in
        setattr(self, f"inp_{name}", obj)

    def expose_output(self, name, obj):
        self.dim_map[f"out_{name}"] = obj.size_out
        setattr(self, f"out_{name}", obj)

    def add_module_input(self, src_module, src_output_name, target_obj):
        self.expose_input(f"{src_module}_{src_output_name}", target_obj)

        if src_module not in self.module_conns:
            self.module_conns[src_module] = []

        self.module_conns[src_module].append(src_output_name)

    def get_inp(self, name):
        return getattr(self, f"inp_{name}")

    def get_out(self, name):
        return getattr(self, f"out_{name}")

    def add_multi_process_support(self):
        with self:
            self.mp_comm_net = self.get_multi_process_node()

    def get_multi_process_hub(self):
        raise NotImplementedError(
            "SpaunModule.get_multi_process_hub needs to be implemented "
            + "by each Spaun module."
        )

    def get_multi_process_node(self):
        self.mp_node = SpaunMPNode(self)

    def setup_inputs_and_outputs(self):
        raise NotImplementedError(
            "SpaunModule.setup_inputs_and_outputs needs to be implemented "
            + "by each Spaun module."
        )

    def setup_spa_inputs_and_outputs(self):
        raise NotImplementedError(
            "SpaunModule.setup_spa_inputs_and_outputs needs to be implemented "
            + "by each Spaun module."
        )

    def setup_module_connections(self, parent_net):
        if hasattr(self, "module_conns"):
            for module, io_names in self.module_conns.items():
                if hasattr(parent_net, module):
                    self.connect_input(parent_net, module, io_names)
                else:
                    warn(f"{self.__class__.__name__} - Could not connect "
                         + f"from '{module}' module.")
        else:
            raise NotImplementedError(
                "SpaunModule.setup_module_connections requires the `module_conns` "
                + "dictionary to be defined by each Spaun module."
            )

    def connect_input(self, parent_net, subnet_name, io_names, synapse=None):
        for io_name in io_names:
            nengo.Connection(
                getattr(parent_net, subnet_name).get_out(io_name),
                self.get_inp(f"{subnet_name}_{io_name}"),
                synapse=synapse
            )


class SpaunMPHub(Module):
    def __init__(self, parent_module):
        self.parent = parent_module
        self.id_str = f"{self.parent.id_str}_hub"
        self.module_conns = self.parent.module_conns

        Module.__init__(self, self.parent.label, self.parent.seed, None)
        self.init_module()
        self.setup_spa_inputs_and_outputs()

    @with_self
    def init_module(self):
        socket = UDPSendReceiveSocket(
            listen_addr=("localhost", self.parent.mp_hub_port),
            remote_addr=("localhost", self.parent.mp_node_port),
            connection_timeout=30000
        )

        self.comm_node = nengo.Node(socket, size_in=0, size_out=0)

        # Replicate the exposed inputs and outputs of the parent module
        d_in = 0
        d_out = 0
        for name, dim in self.parent.dim_map.items():
            io_node = nengo.Node(label=name, size_in=dim, size_out=dim)
            setattr(self, name, io_node)

            if name.startswith("inp_"):
                self.comm_node.size_in += dim
                nengo.Connection(
                    io_node, self.comm_node[d_in:d_in + dim], synapse=0
                )

                d_in += dim

            elif name.startswith("out_"):
                self.comm_node.size_out += dim
                nengo.Connection(
                    self.comm_node[d_out:d_out + dim], io_node, synapse=None
                )

                d_out += dim

    def get_multi_process_hub(self):
        raise RuntimeError(
            "SpaunMPHub.get_multi_process_hub should not be called."
        )

    def get_multi_process_node(self):
        return RuntimeError(
            "SpaunMPHub.get_multi_process_node should not be called."
        )


class SpaunMPNode(Network):
    def __init__(self, parent_module):
        self.parent = parent_module

        Network.__init__(self, "MP_node", self.parent.seed, None)
        self.init_network()

    @with_self
    def init_network(self):
        socket = UDPSendReceiveSocket(
            listen_addr=("localhost", self.parent.mp_node_port),
            remote_addr=("localhost", self.parent.mp_hub_port),
            connection_timeout=30000
        )

        self.comm_node = nengo.Node(socket, size_in=0, size_out=0)

        # Replicate the exposed inputs and outputs of the parent module
        d_in = 0
        d_out = 0
        for name, dim in self.parent.dim_map.items():
            if name.startswith("out_"):
                self.comm_node.size_in += dim
                nengo.Connection(
                    getattr(self.parent, name),
                    self.comm_node[d_in:d_in + dim],
                    synapse=0
                )

                d_in += dim

            elif name.startswith("inp_"):
                self.comm_node.size_out += dim
                nengo.Connection(
                    self.comm_node[d_out:d_out + dim],
                    getattr(self.parent, name),
                    synapse=None
                )

                d_out += dim
