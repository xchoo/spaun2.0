import math
import numpy as np

import nengo
from nengo.spa.vocab import Vocabulary
from nengo.utils.optimization import sp_subvector_optimal_radius

D = 512

vocab = Vocabulary(D)
vocab.parse("A+B")
vocab_keys = sorted(vocab.keys)

print "R1: %f" % (3.5 / math.sqrt(D))
print "R2: %f" % (sp_subvector_optimal_radius(D, 1, 1, 3000))

# print "R3: %f" % (3.5 / math.sqrt(D * 1.5))
# print "R4: %f" % (sp_subvector_optimal_radius(D, 1, 2, 3000))

# def input_func(t):
#     return vocab[vocab_keys[int(t)]].v

# model = nengo.Network(label="test_spaopt")
# with model:
#     inp = nengo.Node(output=input_func)

#     ea1 = nengo.networks.EnsembleArray(ens_dimensions=1, n_neurons=50,
#                                        n_ensembles=D)
#     ea2 = nengo.networks.EnsembleArray(ens_dimensions=1, n_neurons=50,
#                                        n_ensembles=D,
#                                        radius=3.5 / math.sqrt(D))
#     ea3 = nengo.networks.EnsembleArray(ens_dimensions=1, n_neurons=50,
#                                        n_ensembles=D,
#                                        radius=
#                                        sp_subvector_optimal_radius(D, 1,
#                                                                    1, 3000))
#     nengo.Connection(inp, ea1.input)
#     nengo.Connection(inp, ea2.input)
#     nengo.Connection(inp, ea3.input)

#     p1 = nengo.Probe(ea1.output, synapse=0.005)
#     p2 = nengo.Probe(ea2.output, synapse=0.005)
#     p3 = nengo.Probe(ea3.output, synapse=0.005)

# t = 0
# t_step = 0.01
# sim = nengo.Simulator(model)
# while t < len(vocab_keys):
#     print t
#     sim.run(t_step)
#     t += t_step

# import matplotlib.pyplot as plt
# from nengo.spa.utils import similarity
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(sim.trange(), similarity(sim.data, p1, vocab))
# plt.subplot(3, 1, 2)
# plt.plot(sim.trange(), similarity(sim.data, p2, vocab))
# plt.subplot(3, 1, 3)
# plt.plot(sim.trange(), similarity(sim.data, p3, vocab))
# plt.show()
