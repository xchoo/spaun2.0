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

# from .reward import


class RewardEvaluationSystem(Module):
    def __init__(self, label="Reward Evaluation Sys", seed=None,
                 add_to_container=None):
        super(RewardEvaluationSystem, self).__init__(label, seed,
                                                     add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        bias_node = nengo.Node(1)

        # Number of actions in this spaun setup
        num_actions = experiment.num_learn_actions

        # ------------------- Action detection network ------------------------
        # Translates
        self.action_input = nengo.Node(size_in=num_actions)
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
            cfg.make_ens_array(n_ensembles=num_actions)
        nengo.Connection(self.action_input, self.pos_reward_vals.input,
                         transform=np.eye(num_actions), synapse=None)

        # Calculate negative reward values
        self.neg_reward_vals = \
            cfg.make_ens_array(n_ensembles=num_actions)
        nengo.Connection(self.action_input, self.neg_reward_vals.input,
                         transform=np.ones(num_actions) - np.eye(num_actions),
                         synapse=None)

        # Do the appropriate reward cross linking
        for i in range(num_actions):
            # No reward detect --> disinhibit neg_reward_vals
            nengo.Connection(self.reward_detect.output[0],
                             self.neg_reward_vals.ensembles[i].neurons,
                             transform=[[-3]] *
                             self.neg_reward_vals.ensembles[i].n_neurons)
            # Yes reward detect --> disinhibit pos_reward_vals
            nengo.Connection(self.reward_detect.output[1],
                             self.pos_reward_vals.ensembles[i].neurons,
                             transform=[[-3]] *
                             self.pos_reward_vals.ensembles[i].n_neurons)

        # Calculate the utility bias needed (so that the rewards don't send
        # the utilities to +inf, -inf)
        self.util_vals = EnsembleArray(100, num_actions,
                                       encoders=Choice([[1]]),
                                       intercepts=Exponential(0.15, 0.3, 1))
        nengo.Connection(self.reward_detect.output, self.util_vals.input,
                         transform=-np.ones((num_actions, 2)))
        nengo.Connection(self.action_input, self.util_vals.input,
                         transform=np.ones((num_actions, num_actions)),
                         synapse=None)
        nengo.Connection(self.bg_utilities_input,
                         self.util_vals.input, transform=1, synapse=None)

    def setup_connections(self, parent_net, learn_conns=None):
        p_net = parent_net

        # Set up connections from vision module
        if hasattr(p_net, 'vis'):
            nengo.Connection(p_net.vis.output, self.vis_sp_input)
        else:
            warn("RewardEvaluation Module - Cannot connect from 'vis'")

        if hasattr(p_net, 'ps'):
            nengo.Connection(p_net.ps.action, self.action_input,
                             transform=vocab.ps_action.vectors)
        else:
            warn("RewardEvaluation Module - Cannot connect from 'ps'")

        if hasattr(p_net, 'bg'):
            nengo.Connection(p_net.bg.input[:experiment.num_learn_actions],
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
        else:
            warn("RewardEvaluation Module - No learned connections to " +
                 "configure")
