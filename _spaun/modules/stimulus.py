import numpy as np

import nengo
from nengo.processes import PresentInput
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..configurator import cfg
from ..vocabulator import vocab
from ..experimenter import experiment
from .stim import stim_data


# Wrapper function for vision get_image function to pass config rng.
def get_image(label=None):
    return stim_data.get_image(label, cfg.rng)


def get_vocab(label=None):
    if label is None:
        return (np.zeros(vocab.sp_dim), -1)
    if isinstance(label, tuple):
        label = experiment.num_map[label[1]]

    return (vocab.vis_main[str(label)].v, label)


# def stim_func_vis(t):
#     return get_image(experiment.get_stimulus(t))[0]


def stim_func_vis():
    stimulus_list = []

    for stim in experiment.stim_seq_list:
        stimulus_list.append(get_image(stim)[0])

    return PresentInput(np.array(stimulus_list), experiment.present_interval)


def stim_func_vocab(t):
    return get_vocab(experiment.get_stimulus(t))[0]


class SpaunStimulus(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(SpaunStimulus, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        self.log_output = nengo.Node(output=experiment.log_stimulus,
                                     label="Stim Output Logger")
        self.output = nengo.Node(output=stim_func_vis(),
                                 label='Stim Module Out')

        # Normalized output (output values range from 0 to 1)
        self.probe_output = \
            nengo.Node(size_in=stim_data.probe_image_dimensions,
                       label='Stim Module Probe Out')
        nengo.Connection(self.output[stim_data.probe_subsample_inds],
                         self.probe_output,
                         transform=1.0 / stim_data.max_pixel_value,
                         synapse=None)

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


class SpaunInstructionStimulus(Module):
    def __init__(self, label="Instruction Stimulus", seed=None,
                 add_to_container=None):
        super(SpaunInstructionStimulus, self).__init__(label, seed,
                                                       add_to_container)
        self.init_module()

    def get_instr_sp_vec(self, t):
        instr_sps = experiment.get_instruction_sps(t)
        if instr_sps is not None:
            return vocab.parse_instr_sps_list(instr_sps).v
        else:
            return vocab.main.parse('0').v

    @with_self
    def init_module(self):
        self.output = \
            nengo.Node(output=self.get_instr_sp_vec)
