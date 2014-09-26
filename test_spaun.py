import numpy as np
import time

import nengo
from nengo import spa
from nengo.spa.utils import similarity

# ----- Configurations -----
from _spaun.config import cfg
cfg.present_blanks = True
# cfg.use_opencl = False
# cfg.use_opencl = True

# cfg.sp_dim = 256
cfg.max_enum_list_pos = 4
cfg.neuron_type = nengo.LIFRate()


# ----- Spaun imports -----
from _spaun.utils import run_nengo_sim
from _spaun.utils import get_total_n_neurons
from _spaun._vocab.vocabs import vis_vocab
from _spaun._vocab.vocabs import pos_vocab
from _spaun._vocab.vocabs import enum_vocab
from _spaun._vocab.vocabs import task_vocab
from _spaun._vocab.stimulus import stimulus
from _spaun._vocab.stimulus import get_est_runtime
from _spaun._spaun import Vision
from _spaun._spaun import ProdSys
from _spaun._spaun import InfoEnc
from _spaun._spaun import Memory
from _spaun._spaun import InfoDec


# ----- Spaun proper -----
model = spa.SPA(label='Spaun')
with model:
    model.config[nengo.Ensemble].max_rates = cfg.max_rates
    model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
    model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
    model.config[nengo.Connection].synapse = cfg.pstc

    stimulus(model)
    model.vis = Vision()
    model.ps = ProdSys()
    model.enc = InfoEnc()
    model.mem = Memory()
    model.dec = InfoDec()

    actions = spa.Actions(
        'dot(vis, A) --> ps_task = X',
        '0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W',
        'dot(ps_task, W) - dot(vis, QM) --> ps_task = W',
        '0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R',
        'dot(ps_task, R) - dot(vis, QM) --> ps_task = R',
        'dot(vis, QM) - dot(ps_task, W) --> ps_task = DEC',
        '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DECW')
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg, mutual_inhibit=2)

    model.vis.connect_from_stimulus(model.stimulus_node)
    model.ps.connect_from_vision(model.vis)
    model.enc.connect_from_vision(model.vis)
    model.mem.connect_from_vision(model.vis)
    model.mem.connect_from_encoding(model.enc)
    model.dec.connect_from_vision(model.vis)
    model.dec.connect_from_prodsys(model.ps)
    model.dec.connect_from_encoding(model.enc)
    model.dec.connect_from_memory(model.mem)

    p0 = nengo.Probe(model.stimulus_node)
    p1 = nengo.Probe(model.vis.output, synapse=0.005)
    p2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
    # p3 = nengo.Probe(model.vis.blank_detect, synapse=0.005)
    # p4 = nengo.Probe(model.enc.pos_mb.gate, synapse=0.005)
    # p5 = nengo.Probe(model.enc.pos_mb.gateX, synapse=0.005)
    # p6 = nengo.Probe(model.enc.pos_mb.gateN, synapse=0.005)
    # p4 = nengo.Probe(model.mem.mb1.gate, synapse=0.005)
    # p5 = nengo.Probe(model.mem.mb1.gateX, synapse=0.005)
    # p6 = nengo.Probe(model.mem.mb1.gateN, synapse=0.005)
    p4 = nengo.Probe(model.ps.task_mb.gate, synapse=0.005)
    p5 = nengo.Probe(model.ps.task_mb.gateX, synapse=0.005)
    p6 = nengo.Probe(model.ps.task_mb.gateN, synapse=0.005)
    # p7 = nengo.Probe(model.enc.pos_mb.output, synapse=0.005)
    p8 = nengo.Probe(model.mem.mb1, synapse=0.005)
    # p8 = nengo.Probe(model.enc.enc_output, synapse=0.005)
    # p9 = nengo.Probe(model.mem.mb1.gate, synapse=0.005)
    # p10 = nengo.Probe(model.enc.pos_output, synapse=0.005)
    # p11 = nengo.Probe(model.enc.pos_cconv.output, synapse=0.005)
    p12 = nengo.Probe(model.ps.task_mb.output, synapse=0.005)
    p13 = nengo.Probe(model.dec.select_am, synapse=0.005)
    p14 = nengo.Probe(model.dec.select_vis, synapse=0.005)

print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
print "vis n_neurons: %i" % (get_total_n_neurons(model.vis))
print "enc n_neurons: %i" % (get_total_n_neurons(model.enc))
print "mem n_neurons: %i" % (get_total_n_neurons(model.mem))
print "dec n_neurons: %i" % (get_total_n_neurons(model.dec))
print "ps n_neurons: %i" % (get_total_n_neurons(model.ps))
print "bg n_neurons: %i" % (get_total_n_neurons(model.bg))
print "thal n_neurons: %i" % (get_total_n_neurons(model.thal))

# ----- Spaun simulation run -----
print "START BUILD"
timestamp = time.time()

if cfg.use_opencl:
    import pyopencl as cl
    import nengo_ocl

    pltf = cl.get_platforms()[0]
    ctx = cl.Context(pltf.get_devices())
    print "USING OPENCL - devices: %s" % (str(pltf.get_devices()))
    sim = nengo_ocl.sim_ocl.Simulator(model, dt=cfg.sim_dt, context=ctx)
else:
    sim = nengo.Simulator(model, dt=cfg.sim_dt)

t_build = time.time() - timestamp
timestamp = time.time()

print "START SIM - est_runtime: %f" % get_est_runtime()
run_nengo_sim(sim, cfg.sim_dt, get_est_runtime())

t_simrun = time.time() - timestamp
print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build, t_simrun)

# ----- PLOTTING -----
import matplotlib.pyplot as plt
trange = sim.trange()
r = 5
plt.subplot(r, 1, 1)
plt.plot(trange, sim.data[p0])
plt.xlim([trange[0], trange[-1]])

plt.subplot(r, 1, 2)
num_classes = len(vis_vocab.keys)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
                                                            num_classes)])
for i in range(num_classes):
    plt.plot(trange, similarity(sim.data, p1, vis_vocab)[:, i])
# plt.legend(vis_vocab.keys)
plt.xlim([trange[0], trange[-1]])

plt.subplot(r, 1, 3)
plt.plot(trange, sim.data[p2])
plt.xlim([trange[0], trange[-1]])

# plt.subplot(r, 1, 4)
# plt.plot(trange, sim.data[p3])
# plt.xlim([trange[0], trange[-1]])
plt.subplot(r, 1, 4)
plt.plot(trange, sim.data[p13])
plt.plot(trange, sim.data[p14])
plt.xlim([trange[0], trange[-1]])

plt.subplot(r, 1, 5)
plt.plot(trange, sim.data[p4])
plt.plot(trange, sim.data[p5])
plt.plot(trange, sim.data[p6])
plt.xlim([trange[0], trange[-1]])

# plt.subplot(r, 1, 5)
# num_classes = len(pos_vocab.keys)
# colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
#                                                             num_classes)])
# for i in range(num_classes):
#     plt.plot(trange, similarity(sim.data, p7, pos_vocab)[:, i])
# plt.legend(pos_vocab.keys)
# plt.xlim([trange[0], trange[-1]])

# plt.subplot(r, 1, 5)
# plt.plot(trange, sim.data[p9])
# plt.xlim([trange[0], trange[-1]])
# plt.show()

plt.figure()
r = 2

plt.subplot(r, 1, 1)
num_classes = len(enum_vocab.keys)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
                                                            num_classes)])
for i in range(num_classes):
    plt.plot(trange, similarity(sim.data, p8, enum_vocab)[:, i])
# plt.legend(enum_vocab.keys, loc='upper left')
plt.xlim([trange[0], trange[-1]])

# plt.subplot(r, 1, 1)
# num_classes = len(pos_vocab.keys)
# colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
#                                                             num_classes)])
# for i in range(num_classes):
#     plt.plot(trange, similarity(sim.data, p11, pos_vocab)[:, i])
# plt.legend(pos_vocab.keys, loc='upper left')
# plt.xlim([trange[0], trange[-1]])

# plt.subplot(r, 1, 2)
# num_classes = len(pos_vocab.keys)
# colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
#                                                             num_classes)])
# for i in range(num_classes):
#     plt.plot(trange, similarity(sim.data, p10, pos_vocab)[:, i])
# plt.legend(pos_vocab.keys, loc='upper left')
# plt.xlim([trange[0], trange[-1]])

plt.subplot(r, 1, 2)
num_classes = len(task_vocab.keys)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,
                                                            num_classes)])
for i in range(num_classes):
    plt.plot(trange, similarity(sim.data, p12, task_vocab)[:, i])
# plt.legend(task_vocab.keys, loc='upper right')
plt.xlim([trange[0], trange[-1]])

plt.show()
