import nengo_mpi
from nengo import spa
import nengo
from _spaun.config import cfg

from _spaun.modules import Stimulus, Vision

print "Creating graph..."

# b = nx.dorogovtsev_goltsev_mendes_graph(1)
#b = nx.karate_club_graph()
#b = nx.davis_southern_women_graph()
#b = nx.florentine_families_graph()

cfg.use_mpi = True

model = spa.SPA(label='Spaun', seed=cfg.seed)
with model:
    model.config[nengo.Ensemble].max_rates = cfg.max_rates
    model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
    model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
    model.config[nengo.Connection].synapse = cfg.pstc

    model.stim = Stimulus()
    model.vis = Vision()

with model:
    if hasattr(model, 'vis'):
        model.vis.setup_connections(model)

if True:
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)

print "Done creating graph."

G = nengo_mpi.partition.network_to_filter_graph(model)

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)

adjacency_matrix = nx.to_numpy_matrix(G)
plt.imshow(
    adjacency_matrix, cmap="Greys", interpolation="none")
print adjacency_matrix

plt.subplot(2, 1, 2)
order = nx.spectral_ordering(G)
adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=order)
plt.imshow(
    adjacency_matrix, cmap="Greys", interpolation="none")

plt.show()