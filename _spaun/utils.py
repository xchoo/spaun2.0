import numpy as np

from .configurator import cfg
from .experimenter import experiment
from .vocabulator import vocab


def get_total_n_neurons(model):
    return sum([e.n_neurons for e in model.all_ensembles])


def sum_vocab_vecs(vocab, vocab_strs):
    result = vocab[vocab_strs[0]].copy()

    for sp in vocab_strs[1:]:
        result += vocab[sp]

    return result.v


def conf_interval(data, num_samples=5000, confidence=0.95):
    mean_data = np.zeros(num_samples)

    for i in range(num_samples):
        mean_data[i] = np.mean(np.random.choice(data, len(data)))

    mean_data = np.sort(mean_data)

    low_ind = int(num_samples * (1 - confidence) * 0.5)
    high_ind = num_samples - low_ind - 1

    return (np.mean(data), mean_data[low_ind], mean_data[high_ind])


def strs_to_inds(str_list, ref_str_list):
    return [ref_str_list.index(s) for s in str_list]


def str_to_bool(string):
    return string.lower() in ['yes', 'true', 't', '1']


def invol_matrix(dim):
    result = np.eye(dim)
    return result[-np.arange(dim), :]


def get_probe_data_filename(label='probe_data', suffix='', ext='npz'):
    suffix = str(suffix).replace('?', '@')

    raw_seq = experiment.raw_seq_str.replace('?', '@').replace(':', ';')
    raw_seq = raw_seq.replace('>', ')').replace('<', '(')

    if experiment.present_blanks:
        raw_seq = '-'.join(raw_seq)

    return "+".join([label,
                     "_".join([str(type(cfg.neuron_type).__name__),
                               str(vocab.sp_dim)]),
                     raw_seq,
                     str(cfg.seed)]) + \
           ("" if suffix is '' else '(' + suffix + ')') + "." + ext
