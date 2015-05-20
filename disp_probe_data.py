import os
import sys
import numpy as np
import matplotlib.pyplot as plt

supported_data_version = 3

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

if len(sys.argv) > 1:
    data_filename = sys.argv[1]
else:
    data_filename = default_filename

gen_trange = False

if data_filename.endswith('.npz'):
    config_filename = data_filename[:-4] + '_cfg.npz'
    probe_data = np.load(data_filename)

elif data_filename.endswith('.h5'):
    config_dir, filename = os.path.split(data_filename[:-3])
    nameparts = filename.split('+')
    config_filename = os.path.join(config_dir,
                                   '+'.join(nameparts[:2]) + '_cfg.npz')

    import h5py
    probe_data = h5py.File(data_filename)

    gen_trange = True
else:
    raise RuntimeError('File format not supported.')

config_data = np.load(config_filename)

data_version = 0 if 'version' not in config_data.keys() else \
    config_data['version'].item()
if int(data_version) != int(supported_data_version):
    raise Exception('Unsupported data version number. Expected %i, got %i.'
                    % (supported_data_version, data_version))

if not gen_trange:
    trange = probe_data['trange']
else:
    sim_dt = config_data['dt']
    data_len = probe_data[probe_data.keys()[0]].shape[0]
    trange = np.arange(0, data_len * sim_dt, sim_dt)

probe_list = config_data['probe_list']
vocab_dict = config_data['vocab_dict'].item()

print "\nDISPLAYING PROBE DATA. PROBE LIST:"
print probe_list

title_list = [[]]
figs_list = [[]]
for p in probe_list:
    if p == '0':
        figs_list.append([])
        title_list.append([])
    elif p.isdigit() or p[:-1].isdigit():
        figs_list[-1].append(p)
    else:
        title_list[-1].append(p)

for n, fig in enumerate(figs_list):
    plt.figure()

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
                plt.legend(vocab.keys, loc='center left')
        else:
            num_classes = probe_data[probe][-1].size

            if num_classes < 30:
                plt.gca().set_color_cycle([colormap(i) for i in
                                           np.linspace(0, 0.9, num_classes)])
                for i in range(num_classes):
                    plt.plot(trange, probe_data[probe][:, i])
                if disp_legend:
                    plt.legend(map(str, range(num_classes)), loc='center left')
            else:
                plt.plot(trange, probe_data[probe])

        plt.xlim([trange[0], trange[-1]])
        plt.ylabel('%i,%i' % (n + 1, r + 1))

plt.show()

probe_data.close()
