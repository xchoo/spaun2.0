import numpy as np
import nengo
from assoc_mem_2_0 import AssociativeMemory as AM

present_time = 0.5
present_step = 0.1
zero = True
threshold = 0.5


def input_func(t):
    tmp = t / present_time
    if int(tmp) != round(tmp) and zero:
        return 0
    else:
        return int(tmp) * present_step

model = nengo.Network(label='model')

with model:
    in_node = nengo.Node(output=input_func)
    const_node = nengo.Node(output=threshold + 0.5)
    am = AM(np.eye(2), threshold=threshold, wta_output=True,
            wta_inhibit_scale=2.0)

    nengo.Connection(in_node, am.input[0])
    nengo.Connection(const_node, am.input[1])

    probe_in = nengo.Probe(in_node, synapse=None)
    probe_am = nengo.Probe(am.output, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(1 / present_step)

import matplotlib.pyplot as plt

plt.plot(sim.trange(), sim.data[probe_in])
plt.plot(sim.trange(), sim.data[probe_am])
plt.show()
