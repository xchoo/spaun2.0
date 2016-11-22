import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from ..experimenter import experiment
from .vision.data import vis_data


# Wrapper function for vision get_image function to pass config rng.
def get_image(label=None):
    return vis_data.get_image(label, cfg.rng)


def get_vocab(label=None):
    if label is None:
        return (np.zeros(vocab.sp_dim), -1)
    if isinstance(label, tuple):
        label = experiment.num_map[label[1]]

    return (vocab.vis_main[str(label)].v, label)


def stim_func_vis(t):
    return get_image(experiment.get_stimulus(t))[0]


def stim_func_vocab(t):
    return get_vocab(experiment.get_stimulus(t))[0]


class SpaunStimulus(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(SpaunStimulus, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        if cfg.use_mpi:
            import nengo_mpi

            dimension = get_image()[0].size
            self.output = \
                nengo_mpi.SpaunStimulus(dimension, experiment.raw_seq_list,
                                        experiment.present_interval,
                                        experiment.present_blanks)
        else:
            self.output = nengo.Node(output=stim_func_vis,
                                     label='Stim Module Out')

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vocab.vis))


class SpaunStimulusDummy(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(SpaunStimulusDummy, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        dimension = vocab.vis_dim
        self.output = nengo.Node(output=np.random.uniform(size=dimension))

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vocab.vis))
