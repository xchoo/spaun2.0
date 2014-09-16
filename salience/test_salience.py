import nengo
from nengo.utils.distributions import Uniform
from assoc_mem_2_0 import AssociativeMemory as AM

model = nengo.Network(label='model')

n = 50


def input_func(t):
    return t >= 1.0

with model:
    i1 = nengo.Node(output=input_func)

    e1 = nengo.Ensemble(n, 1, max_rates=Uniform(100, 200))
    e2 = nengo.Ensemble(n, 1, max_rates=Uniform(100, 200))

    n1 = nengo.Node(size_in=1)

    am = AM([[1]], threshold=0.3)
    am.add_input('in2', [[1]])

    nengo.Connection(i1, e1, synapse=0.005)
    nengo.Connection(e1, e2, synapse=0.1)
    nengo.Connection(e1, n1, synapse=0.005)
    nengo.Connection(e2, n1, synapse=0.005, transform=-1)
    nengo.Connection(e1, am.input, synapse=0.005)
    nengo.Connection(e2, am.in2, synapse=0.005, transform=-1)

    p1 = nengo.Probe(n1, synapse=0.005)
    p2 = nengo.Probe(am.output, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(2)

import matplotlib.pyplot as plt
plt.plot(sim.trange(), sim.data[p1])
plt.plot(sim.trange(), sim.data[p2])
plt.show()
