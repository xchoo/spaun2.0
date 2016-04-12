import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from _spaun.animation import ArmAnim, DataFunctions, GeneratorFunctions

supported_data_version = 4

# default_filename = 'probe_data_nospc_LIF_256_list3.npz'
# default_filename = 'probe_data_nospc_LIFRate_256.npz'
# default_filename = 'probe_data_nospc_LIF_256.npz'
# default_filename = 'probe_data_nospc_LIF_256_staticIncCConv.npz'
# default_filename = 'probe_data_nospc_LIF_256_testCfgMakes2.npz'
# default_filename = 'probe_data_nospc_LIF_256_testMBCfg3.npz'
# default_filename = 'probe_data_nospc_LIF_256_testDecFR2F.npz'
# default_filename = 'probe_data_spc_LIF.npz'
# default_filename = 'probe_data_nospc_LIF_256_list7.npz'
default_filename = 'probe_data_nospc_LIF_256_list7_B.npz'

# --------------------- PROCESS SYSTEM ARGS ---------------------
if len(sys.argv) > 1:
    data_filename = sys.argv[1]
else:
    data_filename = default_filename

if len(sys.argv) > 2:
    show_grphs = int(sys.argv[2])
    show_anim = int(sys.argv[3])
else:
    show_grphs = True

# --------------------- LOAD SIM DATA ---------------------
gen_trange = False
if data_filename.endswith('.npz'):
    config_filename = data_filename[:-4] + '_cfg.npz'
    probe_data = np.load(data_filename)

elif data_filename.endswith('.h5'):
    # H5 file format (nengo_mpi)
    config_dir, filename = os.path.split(data_filename[:-3])
    nameparts = filename.split('+')
    config_filename = os.path.join(config_dir,
                                   '+'.join(nameparts[:2]) + '_cfg.npz')

    import h5py
    probe_data = h5py.File(data_filename)

    gen_trange = True
else:
    raise RuntimeError('File format not supported.')

# --------------------- LOAD MODEL & PROBE CONFIG DATA ---------------------
config_data = np.load(config_filename)

data_version = 0 if 'version' not in config_data.keys() else \
    config_data['version'].item()
if int(data_version) != int(supported_data_version):
    raise Exception('Unsupported data version number. Expected %i, got %i.'
                    % (supported_data_version, data_version))

vocab_dict = config_data['vocab_dict'].item()

# --------------------- GENERATE T RANGE ---------------------
if not gen_trange:
    trange = probe_data['trange']
else:
    sim_dt = config_data['dt']
    data_len = probe_data[probe_data.keys()[0]].shape[0]
    trange = np.arange(0, data_len * sim_dt, sim_dt)

# --------------------- DISPLAY PROBE DATA ---------------------
print "\nDISPLAYING PROBE DATA."

fig_list = []


# Function to handle closing of one figure window, and close all other figures
def handle_close(fig, fig_list):
    fig_list.pop(fig_list.index(fig))
    if len(fig_list) > 0:
        plt.close(fig_list[0])


# --------------------- DISPLAY GRAPHED DATA ---------------------
if show_grphs:
    graph_list = config_data['graph_list']

    print "GRAPH LIST: "
    print graph_list

    title_list = [[]]
    grph_list = [[]]
    for p in graph_list:
        if p == '0':
            grph_list.append([])
            title_list.append([])
        elif p.isdigit() or p[:-1].isdigit():
            grph_list[-1].append(p)
        else:
            title_list[-1].append(p)

    for n, fig in enumerate(grph_list):
        f = plt.figure()
        f.canvas.mpl_connect('close_event',
                             lambda evt, fig=f, fig_list=fig_list:
                             handle_close(fig, fig_list))
        fig_list.append(f)

        if len(title_list[n]) > 0:
            plt.suptitle(title_list[n][-1])

        max_r = len(fig)
        for r, probe in enumerate(fig):
            disp_legend = False
            if probe[-1] == '*':
                disp_legend = True
                probe = probe[:-1]

            plt.subplot(max_r, 1, r + 1)

            colormap = plt.cm.gist_ncar

            if probe in vocab_dict.keys():
                vocab = vocab_dict[probe]
                num_classes = len(vocab.keys)

                plt.gca().set_color_cycle([colormap(i) for i in
                                           np.linspace(0, 0.9, num_classes)])
                for i in range(num_classes):
                    plt.plot(trange,
                             np.dot(probe_data[probe], vocab.vectors.T)[:, i])
                if disp_legend:
                    plt.legend(vocab.keys, loc='best')
            else:
                num_classes = probe_data[probe][-1].size

                if num_classes < 30:
                    plt.gca().set_color_cycle([colormap(i) for i in
                                               np.linspace(0, 0.9,
                                                           num_classes)])
                    for i in range(num_classes):
                        plt.plot(trange, probe_data[probe][:, i])
                    if disp_legend:
                        plt.legend(map(str, range(num_classes)),
                                   loc='best')
                else:
                    plt.plot(trange, probe_data[probe])

            plt.xlim([trange[0], trange[-1]])
            plt.ylabel('%i,%i' % (n + 1, r + 1))

if show_anim:
    anim_config = config_data['anim_config']

    print "ANIMATION CONFIG: "
    print anim_config

    subplot_width = anim_config[-1]['subplot_width']
    subplot_height = anim_config[-1]['subplot_height']
    max_subplot_cols = anim_config[-1]['max_subplot_cols']

    num_plots = len(anim_config) - 1
    num_cols = num_plots if num_plots < max_subplot_cols else max_subplot_cols
    num_rows = int(np.ceil(1.0 * num_plots / max_subplot_cols))

    # Make the figure to pass to the animation object
    # Note: not hooked into close handler of other figures so that you can
    #       independently close animation figure while keeping others open
    #       (and vice versa)
    f = plt.figure(figsize=(num_cols * subplot_width,
                            num_rows * subplot_height))

    # Make the animation object
    anim_obj = ArmAnim(None, (num_rows, num_cols), f)
    func_map = {}

    # Loop through the animation configuration list and add each subplot
    for i, config in enumerate(anim_config[:-1]):
        # Subplot location
        subplot_row = i / max_subplot_cols
        subplot_col = i % max_subplot_cols

        # Create the data object to use for the animation
        data_func_obj = getattr(DataFunctions, config['data_func'])
        data_func_params = {}
        for param_name in config['data_func_params']:
            data_func_params[param_name] = \
                probe_data[config['data_func_params'][param_name]]
        data_func = data_func_obj(**data_func_params)

        # Add the data function to the function map
        func_map[config['key']] = data_func

        # Add animation subplot to anim_obj
        plot_type_params = dict(config['plot_type_params'])
        plot_type_params.setdefault('key', config['key'])
        plot_type_params.setdefault('tl_loc', (subplot_row, subplot_col))
        getattr(anim_obj, 'add_' + config['plot_type'])(**plot_type_params)

    # Assign the proper data generator function to the animation object and
    # start it
    data_gen_func_params = anim_config[-1]['generator_func_params']
    anim_obj.data_gen_func = \
        lambda: GeneratorFunctions.keyed_data_funcs(trange, func_map,
                                                    **data_gen_func_params)
    anim_obj.start(interval=10)

plt.show()

probe_data.close()
