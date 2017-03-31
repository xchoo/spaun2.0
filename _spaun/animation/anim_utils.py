import time
import sys
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


class GeneratorFunctions(object):
    # @staticmethod
    # def nengo_sim_data(sim, t_stop=None, t_sim_step=None,
    #                    probe_buffer_size=-1):
    #     t = 0.
    #     t_index = 0
    #     dt = sim.dt

    # NOTE: Probe sample interval (s): probe.sample_every
    # NOTE: Get trange for probe: sim.trange(probe.sample_every)

    #     if t_sim_step is None:
    #         t_sim_step = 5 * dt

    #     probe_buffer_size = max(probe_buffer_size, t_sim_step) / dt
    #     t_index_step = t_sim_step / dt

    #     data_dict = {}
    #     source_data_dict = {}

    #     t_data = sim.trange()
    #     t_data_len = len(t_data)
    #     probe_keys = sim.model.probes

    #     for probe in probe_keys:
    #         source_data_dict[probe] = sim.data[probe]

    #     while t < t_stop:
    #         min_t_index = max(0, t_index - probe_buffer_size)

    #         t_index += t_index_step
    #         if t_index >= t_data_len:
    #             t = t_stop
    #         else:
    #             t = t_data[t_index]

    #         for probe in probe_keys:
    #             data_dict[probe] = \
    #                 source_data_dict[probe][min_t_index:t_index + 1, :].T

    #         yield(t_data[min_t_index:t_index + 1], data_dict)
    #         # TODO: Pass all data to animation functions instead of bits of
    #         #       it?

    # @staticmethod
    # def run_nengo_sim_generator(sim, dt, t_stop, t_sim_step=None,
    #                             probe_buffer_size=-1, use_data_dict=True,
    #                             func_list=[], nengo_sim_run_opts=1):
    #     t = 0
    #     if t_sim_step is None:
    #         t_sim_step = 5 * dt

    #     print_progress_bar(t, t_stop)
    #     timestamp_start = time.time()

    #     probe_buffer_size = max(probe_buffer_size, t_sim_step)
    #     num_data_indices = int(np.floor(probe_buffer_size / dt))

    #     data_dict = {}

    #     while t < t_stop:
    #         step_runtime = min(t_sim_step, t_stop - t)
    #         step_runtime = np.ceil(step_runtime / dt) * dt

    #         for func in func_list:
    #             func(t)

    #         if nengo_sim_run_opts:
    #             sim.run(step_runtime, progress_bar=False)
    #         else:
    #             sim.run(step_runtime)
    #         timestamp = time.time()
    #         t = sim.trange()[-1]

    #         if use_data_dict and len(sim.model.probes) > 0:
    #             if num_data_indices > 0:
    #                 data_len = len(sim._probe_outputs[sim.model.probes[0]])

    #                 tdata = sim.trange()[-num_data_indices:]

    #                 data_len_remove = data_len - num_data_indices
    #                 for probe in sim.model.probes:
    #                     if data_len_remove > 0:
    #                         del sim._probe_outputs[probe][:(data_len_remove)]

    #                     pdata = sim.data[probe].T
    #                     data_dict[probe] = pdata
    #             else:
    #                 tdata = sim.trange()

    #                 for probe in sim.model.probes:
    #                     pdata = sim.data[probe].T
    #                     data_dict[probe] = pdata

    #             yield (tdata, data_dict)

    #         print_progress_bar(t, t_stop,
    #                            eta_s=1.0 * (timestamp - timestamp_start)
    #                            * (t_stop - t) / t)
    #     print ""

    @staticmethod
    def keyed_data_funcs(t_data, func_map, t_index_step=1):
        data_dict = {}
        indicies = np.arange(t_data.shape[0])

        for i in indicies[::t_index_step]:
            for key in func_map:
                data_dict[key] = func_map[key](i)
            yield (t_data[i], data_dict)


class DataFunctions(object):
    # NOTE: These functions assume that all probes have same sampling period
    #       see keyed_data_funcs

    @staticmethod
    def arm_path(ee_path_data=None, target_path_data=None,
                 pen_status_data=None, arm_posx_data=None, arm_posy_data=None,
                 arm_pos_bias=None):

        if arm_pos_bias is None:
            arm_pos_bias = [0, 0]

        def data_func(t_index, ee_data=ee_path_data, tgt_data=target_path_data,
                      pen_data=pen_status_data,
                      arm_data=[arm_posx_data, arm_posy_data],
                      arm_bias=arm_pos_bias):
            data = np.zeros(13)

            if ee_data is not None:
                data[0:2] = ee_data[t_index, :]
            else:
                data[0:2] = None

            if tgt_data is not None:
                data[2:4] = tgt_data[t_index, :]
            else:
                data[2:4] = None

            if pen_data is not None:
                data[4] = pen_data[t_index, 0]
            else:
                data[4] = 1

            if arm_data[0] is not None and arm_data[1] is not None:
                data[5:9] = arm_data[0][t_index, :] - arm_bias[0]
                data[9:13] = arm_data[1][t_index, :] - arm_bias[1]

                if ee_data is None:
                    data[0] = arm_data[0][t_index, -1] - arm_bias[0]
                    data[1] = arm_data[1][t_index, -1] - arm_bias[1]
            else:
                data[5:] = None

            return data

        return data_func

    @staticmethod
    def generic_single(data, **args):
        def data_func(t_index, data=data):
            return data[t_index, :]
        return data_func

    @staticmethod
    def generic_constant(data, **args):
        flatten_data = np.asarray(data).flatten()
        print flatten_data.shape

        def data_func(t_index, data=flatten_data):
            return data
        return data_func
