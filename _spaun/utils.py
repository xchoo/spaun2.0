import sys
import time
import numpy as np


def print_progress_bar(t, t_max, steps=10, eta_s=None):
    percent_done = min(t / t_max * 1.0, 1)
    percent_per_bar = steps / 100.0
    bars_filled = int(round(percent_done / percent_per_bar, 2))

    # frac_percent = (percent_done * 10) - int(percent_done * 10)

    eta_str = "" if eta_s is None else \
        time.strftime("%Hh %Mm %Ss", time.gmtime(max(eta_s, 0)))

    sys.stdout.write("\r[" + "#" * bars_filled +
                     "_" * (steps - bars_filled) +
                     "]" + "% 7.2f%% | t = %.2f / %.2fs | etc: %s" %
                     (percent_done * 100.0, t, t_max, eta_str))
    sys.stdout.flush()


def run_nengo_sim_generator(sim, dt, t_stop, t_sim_step=None,
                            probe_buffer_size=-1, use_data_dict=True):
    t = 0
    if t_sim_step is None:
        t_sim_step = 5 * dt

    print_progress_bar(t, t_stop)
    timestamp_start = time.time()

    probe_buffer_size = max(probe_buffer_size, t_sim_step)
    num_data_indices = int(np.floor(probe_buffer_size / dt))

    data_dict = {}

    while t < t_stop:
        step_runtime = min(t_sim_step, t_stop - t)
        step_runtime = np.ceil(step_runtime / dt) * dt

        sim.run(step_runtime)
        timestamp = time.time()
        t = sim.trange()[-1]

        if use_data_dict and len(sim.model.probes) > 0:
            if num_data_indices > 0:
                data_len = len(sim._probe_outputs[sim.model.probes[0]])

                ## Pad the data with zeros
                # if data_len < num_data_indices:
                #     num_to_fill = num_data_indices - data_len

                #     tdata = np.append(sim.trange(),
                #                       np.linspace(t + dt,
                #                                   probe_buffer_size,
                #                                   num_to_fill))
                #     for probe in sim.model.probes:
                #         pdata_dims = sim._probe_outputs[probe][-1].shape[0]
                #         pdata = np.append(sim.data[probe],
                #                           [[None] * pdata_dims] * num_to_fill,
                #                           axis=0).T
                #         data_dict[probe] = pdata
                # else:
                tdata = sim.trange()[-num_data_indices:]

                data_len_remove = data_len - num_data_indices
                for probe in sim.model.probes:
                    if data_len_remove > 0:
                        del sim._probe_outputs[probe][:(data_len_remove)]

                    pdata = sim.data[probe].T
                    data_dict[probe] = pdata
            else:
                tdata = sim.trange()

                for probe in sim.model.probes:
                    pdata = sim.data[probe].T
                    data_dict[probe] = pdata

            yield (tdata, data_dict)

        print_progress_bar(t, t_stop, eta_s=1.0 * (timestamp - timestamp_start)
                           * (t_stop - t) / t)
    print ""


def run_nengo_sim(sim, dt, t_stop, t_sim_step=None):
    for _ in run_nengo_sim_generator(sim, dt, t_stop, t_sim_step,
                                     use_data_dict=False):
        pass


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
