import os
import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import vis_vocab
from ..vision import get_image as vis_get_image


num_map = {'0': 'ZER', '1': 'ONE', '2': 'TWO', '3': 'THR', '4': 'FOR',
           '5': 'FIV', '6': 'SIX', '7': 'SEV', '8': 'EIG', '9': 'NIN'}
sym_map = {'[': 'OPEN', ']': 'CLOSE', '?': 'QM'}


# Wrapper function for vision get_image function to pass config rng.
def get_image(label=None):
    return vis_get_image(label, cfg.rng)


def get_vocab(label=None):
    if label is None:
        return (np.zeros(cfg.sp_dim), -1)
    if isinstance(label, tuple):
        label = num_map[label[1]]

    return (vis_vocab[str(label)].v, 0)


def insert_mtr_wait_sym(num_mtr_responses):
    # Add 0.5 second motor response minimum
    num_mtr_responses += 0.5
    est_mtr_response_time = num_mtr_responses * cfg.mtr_est_digit_response_time
    extra_spaces = int(est_mtr_response_time / (cfg.present_interval * 2 **
                                                cfg.present_blanks))

    cfg.raw_seq.extend([None] * extra_spaces)
    cfg.stim_seq.extend([None] * extra_spaces)


def parse_raw_seq():
    raw_seq = list(cfg.raw_seq_str)
    hw_num = False  # Flag to indicate to use a hand written number
    prev_c = ''

    cfg.raw_seq = []
    cfg.stim_seq = []

    num_mtr_responses = 0.0

    for c in raw_seq:
        if c in sym_map:
            c = sym_map[c]
        if not hw_num and c in num_map:
            c = num_map[c]
        if c == '#':
            hw_num = True
            continue
        if c == 'X':
            num_mtr_responses += 1
            continue
        elif num_mtr_responses > 0:
            insert_mtr_wait_sym(num_mtr_responses)
            num_mtr_responses = 0

        cfg.raw_seq.append(c)

        # If previous character is identical to current character, insert a
        # space between them.
        if prev_c == c:
            cfg.stim_seq.append('.')

        if c.isdigit():
            cfg.stim_seq.append((get_image(c)[1], c))
            hw_num = False
        else:
            cfg.stim_seq.append(c)

        prev_c = c

    # Insert trailing motor response wait symbols
    insert_mtr_wait_sym(num_mtr_responses)


def stim_func(t, stim_seq=None, get_func=None):
    ind = t / cfg.present_interval / (2 ** cfg.present_blanks)

    if (cfg.present_blanks and int(ind) != int(round(ind))) or \
       int(ind) >= len(stim_seq) or stim_seq[int(ind)] == '.':
        image_data = get_func()
    else:
        image_data = get_func(stim_seq[int(ind)])

    return image_data[0]


def stim_func_vis(t):
    return stim_func(t, cfg.stim_seq, get_image)


def stim_func_vocab(t):
    return stim_func(t, cfg.stim_seq, get_vocab)


def get_est_runtime():
    return len(cfg.stim_seq) * cfg.present_interval * (2 ** cfg.present_blanks)


class Stimulus(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(Stimulus, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        if cfg.use_mpi:
            import nengo_mpi

            dimension = get_image()[0].size
            self.output = \
                nengo_mpi.SpaunStimulus(dimension, cfg.raw_seq,
                                        cfg.present_interval,
                                        cfg.present_blanks)
        else:
            self.output = nengo.Node(output=stim_func_vis,
                                     label='Stim Module Out')

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))


class StimulusDummy(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(StimulusDummy, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        dimension = 28 * 28
        self.output = nengo.Node(output=np.random.uniform(size=dimension))

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))


def monitor_func(t, monitor_data_obj, stim_seq=None):
    ind = t / cfg.present_interval / (2 ** cfg.present_blanks)

    if (cfg.present_blanks and int(ind) != int(round(ind))) or \
       int(ind) >= len(stim_seq) or stim_seq[int(ind)] == '.':
        pass
    else:
        pass


class MonitorData(object):
    def __init__(self):
        self.prev_ind = -1
        self.data_filename = \
            os.path.join(cfg.data_dir,
                         cfg.get_probe_data_filename(suffix='log', ext='txt'))
        self.data_obj = open(self.data_filename, 'a')

    def close_data_obj(self):
        self.data_obj.close()


class Monitor(Module):
    def __init__(self, label="Monitor", seed=None, add_to_container=None):
        super(Monitor, self).__init__(label, seed, add_to_container)
        self.monitor_data = MonitorData()
        self.init_module()

    @with_self
    def init_module(self):
        if cfg.use_mpi:
            raise RuntimeError('Not Implemented')
        else:
            self.output = nengo.Node(output=monitor_func,
                                     label='Stim Module Out')

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))

    def close(self):
        self.monitor_data.close_data_obj()
