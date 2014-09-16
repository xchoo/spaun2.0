import numpy as np
import bisect as bs
import matplotlib.pyplot as plt

# data = np.load('SPs_SYM.npz')
mnist = np.load('mnist.npz')

labels = mnist['labels']
labels_unq = np.unique(labels)
label_inds = []
label_cnts = []
for c in labels_unq:
    label_inds.append(range(bs.bisect_left(labels, c),
                            bs.bisect_right(labels, c)))
    label_cnts.append(len(label_inds[-1]))

# for key in data.keys():
#     means = [0] * len(labels_unq)
#     for c in range(len(labels_unq)):
#         means[c] = sum(np.array(data[key][label_inds[c]])) / len(label_inds[c])
#         mean_mag = np.linalg.norm(means[c])
#         means[c] /= mean_mag
#         # means[c] *= 0.8 / mean_mag
#         means[c] *= 1.0 / mean_mag
#         print np.linalg.norm(means[c])
#     means = np.matrix(means)
#     sims = np.matrix(data[key]) * means.T

#     np.savez('means_%s.npz' % key, means=means)

#     colormap = plt.cm.gist_ncar
#     plt.gca().set_color_cycle([colormap(i) for i in
#                                np.linspace(0, 0.9, len(labels_unq))])

#     for c in range(len(labels_unq)):
#         plt.plot(range(len(labels)), sims[:, c])
#     plt.title(key)
#     plt.tight_layout()
#     plt.legend(labels_unq, loc='lower right')
#     plt.show()
#     break

data = np.load('class_10.npz')['class_10']
data3 = []
data2 = [0] * len(labels_unq)

for e, d in enumerate(data):
    data3.append(data[e][labels[e]])
    data2[labels[e]] += data[e][labels[e]]

data2 = np.array(data2)
data3 = np.array(data3)
means = []

for n in range(len(labels_unq)):
    print max(data3[label_inds[n]]), min(data3[label_inds[n]]), \
        np.mean(data3[label_inds[n]])
    means.append(np.mean(data3[label_inds[n]]))

print label_cnts
print np.divide(data2, label_cnts)
# print summed

means = np.array(means)
means = means / max(means)
means = 1.0 / means
print means

np.savez('scales_200D.npz', scales=means)
