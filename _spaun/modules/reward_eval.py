import numpy as np
from warnings import warn

import nengo
from nengo.dists import Choice, Exponential
from nengo.networks import EnsembleArray
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from ..experimenter import experiment

from .spaun_module import SpaunModule, SpaunMPHub


class RewardEvaluationSystem(SpaunModule):
    def __init__(self, label="Reward Evaluation Sys", seed=None,
                 add_to_container=None):

        module_id_str = "reward"
        module_ind_num = 12

        super(RewardEvaluationSystem, self).__init__(
            module_id_str, module_ind_num, label, seed, add_to_container
        )

    @with_self
    def init_module(self):
        super().init_module()

        # Number of actions in this spaun setup
        num_actions = experiment.num_learn_actions

        # --------------------------- Bias nodes ---------------------------- #
        bias_node = nengo.Node(1, label="Bias")

        # ------------------- Action detection network ------------------------
        # Translates action semantic pointers (from production system) into
        # array of 1's and 0's
        self.actions = cfg.make_thresh_ens_net(num_ens=num_actions)
        self.action_input = self.actions.input
        self.bg_utilities_input = nengo.Node(size_in=num_actions)
        self.vis_sp_input = nengo.Node(size_in=vocab.sp_dim)

        # ------------------- Reward detection network ------------------------
        # Translates visual input into reward yes/no signals
        # Note: Output of reward_detect is inverted
        num_reward_sps = len(vocab.reward.keys)
        self.reward_detect = cfg.make_thresh_ens_net(num_ens=num_reward_sps)
        nengo.Connection(bias_node, self.reward_detect.input,
                         transform=np.ones(num_reward_sps)[:, None])
        nengo.Connection(self.vis_sp_input, self.reward_detect.input,
                         transform=-vocab.reward.vectors, synapse=None)

        # Calculate positive reward values
        self.pos_reward_vals = \
            cfg.make_ens_array(n_ensembles=num_actions, ens_dimensions=1,
                               radius=1)
        nengo.Connection(self.actions.output, self.pos_reward_vals.input,
                         transform=np.eye(num_actions))

        # Calculate negative reward values
        self.neg_reward_vals = \
            cfg.make_ens_array(n_ensembles=num_actions, ens_dimensions=1,
                               radius=1)
        nengo.Connection(self.actions.output, self.neg_reward_vals.input,
                         transform=np.ones(num_actions) - np.eye(num_actions))

        # Do the appropriate reward cross linking
        for i in range(num_actions):
            # No reward detect --> disinhibit neg_reward_vals
            nengo.Connection(self.reward_detect.output[0],
                             self.neg_reward_vals.ensembles[i].neurons,
                             transform=[[-5]] *
                             self.neg_reward_vals.ensembles[i].n_neurons)
            # Yes reward detect --> disinhibit pos_reward_vals
            nengo.Connection(self.reward_detect.output[1],
                             self.pos_reward_vals.ensembles[i].neurons,
                             transform=[[-5]] *
                             self.pos_reward_vals.ensembles[i].n_neurons)

        # Calculate the utility bias needed (so that the rewards don't send
        # the utilities to +inf, -inf)
        self.util_vals = \
            EnsembleArray(100, num_actions, encoders=Choice([[1]]),
                          intercepts=Exponential(0.15, cfg.learn_util_min, 1))
        nengo.Connection(self.reward_detect.output, self.util_vals.input,
                         transform=-np.ones((num_actions, 2)))
        nengo.Connection(self.actions.output, self.util_vals.input,
                         transform=np.ones((num_actions, num_actions)))
        nengo.Connection(self.bg_utilities_input,
                         self.util_vals.input, transform=1, synapse=None)

        # ################ DEBUG node for computed reward values ################
        self.reward_node = nengo.Node(size_in=num_actions)

    @with_self
    def setup_inputs_and_outputs(self):
        # ------ Expose inputs for external connections ------
        # Set up connections from vision module
        if cfg.has_vis:
            self.add_module_input("vis", "main", vocab.sp_dim)

            nengo.Connection(self.get_inp("vis_main"), self.vis_sp_input)

        if cfg.has_ps:
            self.add_module_input("ps", "action", vocab.ps_action.dimensions)
            nengo.Connection(self.get_inp("ps_action"), self.action_input,
                             transform=vocab.ps_action.vectors)

    def setup_spa_inputs_and_outputs(self):
        pass

    def setup_module_connections(self, parent_net, learn_conns=None):
        super(RewardEvaluationSystem, self).setup_module_connections(parent_net)

        if hasattr(parent_net, "bg"):
            nengo.Connection(parent_net.bg.input[:experiment.num_learn_actions],
                             self.bg_utilities_input)

        # Configure learning rules on learned connections
        if learn_conns is not None:
            for i, conn in enumerate(learn_conns):
                conn.learning_rule_type = nengo.PES(cfg.learn_learning_rate)
                nengo.Connection(self.pos_reward_vals.output[i],
                                 conn.learning_rule, transform=-1)
                nengo.Connection(self.neg_reward_vals.output[i],
                                 conn.learning_rule, transform=-1)
                nengo.Connection(self.util_vals.output[i],
                                 conn.learning_rule, transform=1)

                # ################ DEBUG connections ################
                nengo.Connection(self.pos_reward_vals.output[i],
                                 self.reward_node[i], transform=-1)
                nengo.Connection(self.neg_reward_vals.output[i],
                                 self.reward_node[i], transform=-1)
                nengo.Connection(self.util_vals.output[i],
                                 self.reward_node[i], transform=1)
                # ################ DEBUG connections ################
        else:
            warn("RewardEvaluation Module - No learned connections to " +
                 "configure")

    def get_multi_process_hub(self):
        raise RuntimeError("Reward module must be built as part of the " +
                           "main Spaun network, and does not support " +
                           "creation as a separate Spaun process.")

