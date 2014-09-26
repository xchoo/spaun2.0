import nengo
from nengo.spa.module import Module

from ..config import cfg
from ..vocabs import vis_vocab
from ..vision import get_image

# stim_seq = ['A', 'ZER', 'OPEN', get_image('4')[1], 'CLOSE', 'QM', None, None,
#             'A', 'ONE', 'OPEN', get_image('1')[1], get_image('2')[1],
#             get_image('3')[1], 'CLOSE', 'QM', None, None]
stim_seq = ['A', 'ZER', 'OPEN', get_image('4')[1], 'CLOSE', 'QM', None]


def stim_func(t):
    ind = t / cfg.present_interval / (2 ** cfg.present_blanks)

    if cfg.present_blanks and int(ind) != int(round(ind)):
        image_data = get_image()
    else:
        image_data = get_image(stim_seq[int(ind)])

    return image_data[0]


def get_est_runtime():
    return len(stim_seq) * cfg.present_interval * (2 ** cfg.present_blanks)


class Stimulus(Module):
    def __init__(self):
        super(Stimulus, self).__init__()

        self.output = nengo.Node(output=stim_func)

        # Define vocabulary inputs and outputs
        self.outputs = dict(default=(self.output, vis_vocab))
