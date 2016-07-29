import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from _spaun.animation import ArmAnim, DataFunctions, GeneratorFunctions
from _spaun.modules.vision.data import VisionDataObject

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
    data_filename = sys.argv[1].strip()
    data_filename = data_filename.replace('"', '')
else:
    data_filename = default_filename

if len(sys.argv) > 2:
    show_grphs = int(sys.argv[2])
    show_anim = int(sys.argv[3])
    show_io = int(sys.argv[4])
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
    raise RuntimeError('Filename: %s - File format not supported.' %
                       data_filename)

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


# Helper function to calculate differences in images (for show_io)
def rmse(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

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

            # Compress plots
            f.subplots_adjust(hspace=0.05, bottom=0.05, left=0.1, right=0.98,
                              top=0.95)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

if show_anim or show_io:
    anim_config = config_data['anim_config']

    print "ANIMATION CONFIG: "
    print anim_config

if show_anim:
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

if show_io:
    vis_stim_config = anim_config[0]
    vis_stim_probe_id_str = vis_stim_config['data_func_params']['data']
    vis_stim_data = np.array(probe_data[vis_stim_probe_id_str])

    arm_data_dict = anim_config[1]['data_func_params']
    ee_probe_id_str = arm_data_dict['ee_path_data']
    ee_data = np.array(probe_data[ee_probe_id_str])
    pen_probe_id_str = arm_data_dict['pen_status_data']
    pen_data = np.array(probe_data[pen_probe_id_str])

    arm_data_scale = anim_config[1]['plot_type_params']['xlim'][1]

    A_img = VisionDataObject().get_image('A')[0]
    num_cols = 0
    curr_col_ind = 0

    plot_data = []
    plot_type = []

    pen_down = False
    pen_down_ind = -1

    img_ind_filter = []
    path_len_filter = 200

    prev_img = np.zeros(vis_stim_data.shape[1])
    for i in range(vis_stim_data.shape[0]):
        img = vis_stim_data[i, :]

        if rmse(prev_img, img) > 0.1:
            # Img data is an 'A', so reset things
            if rmse(img, A_img) < 0.1:
                if len(plot_data) > 0:
                    num_cols = max(num_cols, len(plot_data[-1]))
                plot_data.append([])
                plot_type.append([])
                curr_col_ind = 0
            if len(plot_data) <= 0:
                plot_data.append([])
                plot_type.append([])
            if (curr_col_ind in img_ind_filter) or len(img_ind_filter) == 0:
                plot_data[-1].append(np.array(img))
                plot_type[-1].append("im")
            prev_img = img
            curr_col_ind += 1

        if not pen_down and pen_data[i] > 0.5:
            pen_down = True
            pen_down_ind = i
        elif pen_down and pen_data[i] < 0.25:
            pen_down = False
            path_data = ee_data[pen_down_ind:i, :]
            if path_data.shape[0] > path_len_filter:
                if len(plot_data) <= 0:
                    plot_data.append([])
                    plot_type.append([])
                plot_data[-1].append(path_data)
                plot_type[-1].append("path")

    # Get number of columns (gotta do this here to take into account last row)
    # added to the plot_data array
    num_cols = max(num_cols, len(plot_data[-1]))

    plt.figure(figsize=(min(2 * num_cols, 18), min(2 * len(plot_data), 12)))
    for i in range(len(plot_data)):
        for j in range(len(plot_data[i])):
            plt.subplot(len(plot_data), num_cols, i * num_cols + j + 1,
                        aspect=1)
            if plot_type[i][j] == 'im':
                # Reshape to 28 * 28 (TODO: FIX for generic images?)
                im_data = plot_data[i][j].reshape((28, 28))
                plt.imshow(im_data, cmap=plt.get_cmap('gray'))
                plt.xticks([])
                plt.yticks([])
            else:
                plt.plot(plot_data[i][j][:, 0], plot_data[i][j][:, 1])
                plt.xticks([])
                plt.yticks([])
                plt.xlim(-arm_data_scale, arm_data_scale)
                plt.ylim(-arm_data_scale, arm_data_scale)
    plt.tight_layout()

plt.show()
probe_data.close()
