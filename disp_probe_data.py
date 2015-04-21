import sys
import numpy as np
import matplotlib.pyplot as plt

supported_data_version = 1

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

config_filename = data_filename[:-4] + '_cfg.npz'

probe_data = np.load(data_filename)
config_data = np.load(config_filename)

data_version = 0 if 'version' not in config_data.keys() else \
    config_data['version'].item()
if int(data_version) != int(supported_data_version):
    raise Exception('Unsupported data version number. Expected %i, got %i.'
                    % (supported_data_version, data_version))

trange = probe_data['trange']
probe_list = config_data['probe_list']
vocab_dict = config_data['vocab_dict'].item()

# probe_list = np.array(probe_list)
print "\nDISPLAYING PROBE DATA. PROBE LIST:"
print probe_list

num_rows = np.where(probe_list == '0')[0]
num_rows = np.append(num_rows, probe_list.shape[0])

max_r_ind = 0
max_r = -1
r = 1
for probe in probe_list:
    disp_legend = False
    if probe[-1] == "*":
        disp_legend = True
        probe = probe[:-1]

    if probe == '0':
        r = 1
        max_r = num_rows[max_r_ind]
        max_r_ind += 1
        plt.figure()
    else:
        colormap = plt.cm.gist_ncar

        plt.subplot(num_rows[max_r_ind] - max_r - 1, 1, r)
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
            num_classes = probe_data[probe][-1].shape[0]

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
        plt.ylabel('%i,%i' % (max_r_ind + 1, r))
        r += 1
plt.show()
