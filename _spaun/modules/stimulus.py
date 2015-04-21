import nengo
from nengo.spa.module import Module
from nengo.utils.network import with_self

from ..config import cfg
from ..vocabs import vis_vocab
from ..vision import get_image as vis_get_image


# Wrapper function for vision get_image function to pass config rng.
def get_image(label=None):
    return vis_get_image(label, cfg.rng)

# #### DEFINE STIMULUS SEQUENCE ##########
# Todo: Make this customizable.

num_mtr_responses = 1

stim_seq = ['A']
num_mtr_responses = 0.4
# num_mtr_responses = 0

# stim_seq = ['A', 'ZER', 'OPEN', get_image('1')[1], get_image('2')[1],
#             get_image('3')[1], 'CLOSE', 'QM']
# num_mtr_responses = 4
# num_mtr_responses = 0.4

# stim_seq = ['A', 'ZER', 'OPEN', get_image('4')[1], 'CLOSE', 'QM']
# num_mtr_responses = 2

# stim_seq = ['A', 'ONE', 'OPEN', get_image('4')[1], 'CLOSE', 'QM']
# num_mtr_responses = 2

# stim_seq = ['A', 'THR', 'OPEN', get_image('1')[1], get_image('2')[1],
#             get_image('3')[1], get_image('4')[1], get_image('5')[1],
#             get_image('1')[1], get_image('2')[1], 'CLOSE', 'QM']
# num_mtr_responses = 8

# stim_seq = ['A', 'FOR', 'OPEN', get_image('1')[1], 'CLOSE',
#             'OPEN', get_image('4')[1], 'CLOSE', 'QM']
# num_mtr_responses = 6

# stim_seq = ['A', 'FIV', 'OPEN', get_image('1')[1], get_image('2')[1],
#             get_image('3')[1], 'M', 'OPEN',
#             get_image('1')[1], 'CLOSE', 'QM']
# num_mtr_responses = 2

# stim_seq = ['A', 'SIX',
#             'OPEN', get_image('1')[1], get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('8')[1], get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('4')[1], get_image('2')[1], 'CLOSE', 'QM']
# num_mtr_responses = 2

# stim_seq = ['A', 'SEV',
#             'OPEN', get_image('1')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('3')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('3')[1], 'CLOSE',
#             'OPEN', get_image('4')[1], 'CLOSE',
#             'OPEN', get_image('3')[1], 'CLOSE',
#             'OPEN', get_image('4')[1], 'CLOSE', 'QM']
# stim_seq = ['A', 'SEV',
#             'OPEN', get_image('1')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE',
#             'OPEN', get_image('3')[1], 'CLOSE',
#             'OPEN', get_image('2')[1], 'CLOSE', 'QM']
# num_mtr_responses = 2

est_mtr_response_time = num_mtr_responses * cfg.mtr_est_digit_response_time
extra_spaces = int(est_mtr_response_time / (cfg.present_interval * 2 **
                                            cfg.present_blanks))

stim_seq.extend([None] * extra_spaces)

# stim_seq.extend(['A', 'ZER'])

# stim_seq.extend(['A', 'ZER', 'OPEN', get_image('1')[1], 'CLOSE', 'QM'])
# num_mtr_responses = 1
# est_mtr_response_time = num_mtr_responses * cfg.mtr_est_digit_response_time
# extra_spaces = int(est_mtr_response_time / (cfg.present_interval * 2 **
#                                             cfg.present_blanks))

# stim_seq.extend([None] * extra_spaces)


def stim_func(t):
    ind = t / cfg.present_interval / (2 ** cfg.present_blanks)

    if (cfg.present_blanks and int(ind) != int(round(ind))) or \
       int(ind) >= len(stim_seq):
        image_data = get_image()
    else:
        image_data = get_image(stim_seq[int(ind)])

    return image_data[0]


def get_est_runtime():
    return len(stim_seq) * cfg.present_interval * (2 ** cfg.present_blanks)


class Stimulus(Module):
    def __init__(self, label="Stimulus", seed=None, add_to_container=None):
        super(Stimulus, self).__init__(label, seed, add_to_container)
        self.init_module()

    @with_self
    def init_module(self):
        self.output = nengo.Node(output=stim_func, label='Stim Module Out')

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))
