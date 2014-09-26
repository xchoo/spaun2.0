import sys


def print_progress_bar(t, t_max, steps=10):
    percent_done = t / t_max * 1.0
    percent_per_bar = steps / 100.0
    bars_filled = int(round(percent_done / percent_per_bar, 2))
    sys.stdout.write("\r[" + "#" * bars_filled + "_" * (steps - bars_filled) +
                     "]" + " % 6.2f%% | t = %.2fs" % (percent_done * 100.0,
                                                        t))
    sys.stdout.flush()


def run_nengo_sim(sim, dt, t_stop, t_sim_step=None):
    t = 0
    if t_sim_step is None:
        t_sim_step = 5 * dt

    print_progress_bar(t, t_stop)
    while t < t_stop:
        step_runtime = min(t_sim_step, t_stop - t)
        sim.run(step_runtime)
        t += step_runtime
        print_progress_bar(t, t_stop)
    print ""


def get_total_n_neurons(model):
    return sum([e.n_neurons for e in model.all_ensembles])


def sum_vocab_vecs(vocab, vocab_strs):
    result = vocab[vocab_strs[0]]

    for str in vocab_strs[1:]:
        result += vocab[str]

    return result.v
