import numpy as np
import time
import sys

import nengo
from nengo import spa

# ----- Configurations -----
from _spaun.config import cfg
# cfg.present_blanks = True
if len(sys.argv) > 1:
    cfg.backend = sys.argv[1]

cfg.sp_dim = 256
# cfg.sp_dim = 512
# cfg.sp_dim = 16
# cfg.sp_dim = 64
# cfg.max_enum_list_pos = 4
# cfg.neuron_type = nengo.LIFRate()
# cfg.gen_probe_data_filename("directItemCConvStaticIncCConv")
# cfg.gen_probe_data_filename("staticIncCConv")
# cfg.gen_probe_data_filename("testDecFR2")
# cfg.gen_probe_data_filename("list7")
cfg.gen_probe_data_filename()

# make_probes = False
make_probes = True

# ----- Seeeeeeeed -----
# cfg.set_seed(1413987955)
# cfg.set_seed(1414248095)
# cfg.set_seed(1429562767)
print "MODEL SEED: %i" % cfg.seed
print "BACKEND: %s" % cfg.backend.upper()

# ----- Spaun imports -----
from _spaun.utils import run_nengo_sim
from _spaun.utils import get_total_n_neurons
from _spaun.probes import idstr, config_and_setup_probes
from _spaun.vocabs import vocab
from _spaun.modules import get_est_runtime
from _spaun.modules import Stimulus, Vision, ProdSys, InfoEnc, Memory, InfoDec
from _spaun.modules import Motor
from _spaun.modules.stimulus import stim_seq

print "PROBE FILENAME: %s" % cfg.probe_data_filename

# ----- Spaun proper -----
model = spa.SPA(label='Spaun', seed=cfg.seed)
with model:
    model.config[nengo.Ensemble].max_rates = cfg.max_rates
    model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
    model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
    model.config[nengo.Connection].synapse = cfg.pstc

    model.stim = Stimulus()
    model.vis = Vision()
    model.ps = ProdSys()
    model.enc = InfoEnc()
    model.mem = Memory()
    model.dec = InfoDec()
    model.mtr = Motor()

    model.test = spa.Buffer(cfg.sp_dim)

    if hasattr(model, 'vis') and hasattr(model, 'ps'):
        copy_draw_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W, ps_state = TRANS0',  # noqa
             'dot(ps_task, W) - dot(vis, QM) --> ps_task = W, ps_state = ps_state']  # noqa
        recog_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R, ps_state = TRANS0',  # noqa
             'dot(ps_task, R) - dot(vis, QM) --> ps_task = R, ps_state = ps_state']  # noqa
        mem_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, THR)) --> ps_task = M, ps_state = TRANS0',  # noqa
             'dot(ps_task, M) - 0.5 * dot(vis, F + R + QM) --> ps_task = M, ps_state = ps_state',  # noqa
             '0.5 * (dot(ps_task, M) + dot(vis, F)) --> ps_task = M, ps_dec = FWD',  # noqa
             '0.5 * (dot(ps_task, M) + dot(vis, R)) --> ps_task = M, ps_dec = REV']  # noqa
        count_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, FOR)) --> ps_task = C, ps_state = TRANS0',  # noqa
             '0.5 * (dot(ps_task, C) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = C, ps_state = TRANS1',  # noqa
             '0.5 * (dot(ps_task, C) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = C, ps_state = CNT']  # noqa
        # Count action is incomplete!
        qa_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, FIV)) --> ps_task = A, ps_state = TRANS0',  # noqa
             'dot(ps_task, A) - 0.5 * dot(vis, M + P + QM) --> ps_task = A, ps_state = ps_state',  # noqa
             '0.5 * (dot(ps_task, A) + dot(vis, M)) --> ps_task = M, ps_state = QAN',  # noqa
             '0.5 * (dot(ps_task, A) + dot(vis, P)) --> ps_task = M, ps_state = QAP']  # noqa
        rvc_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, SIX)) --> ps_task = V, ps_state = TRANS0',  # noqa
             '0.5 * (dot(ps_task, V) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = V, ps_state = TRANS1',  # noqa
             '0.5 * (dot(ps_task, V) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = V, ps_state = TRANS0']  # noqa
        fi_action = \
            ['0.5 * (dot(ps_task, X) + dot(vis, SEV)) --> ps_task = F, ps_state = TRANS0',  # noqa
             '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS1',  # noqa
             '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS2',  # noqa
             '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS2)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS0']  # noqa
        decode_action = \
            ['dot(vis, QM) - 0.5 * dot(ps_task, W + C) --> ps_task = DEC, ps_dec = ps_dec, ps_state = ps_state',  # noqa
             '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DECW, ps_dec = ps_dec, ps_state = ps_state',  # noqa
             '0.5 * (dot(vis, QM) + dot(ps_task, C)) --> ps_task = DECC, ps_dec = ps_dec, ps_state = ps_state']  # noqa
        default_action = \
            ['0.5 --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']

        all_actions = (copy_draw_action + recog_action +
                       mem_action + count_action + qa_action + rvc_action +
                       fi_action + decode_action + default_action)

        actions = spa.Actions(*all_actions)
        model.bg = spa.BasalGanglia(actions=actions)
        model.thal = spa.Thalamus(model.bg, mutual_inhibit=2)

# ----- Set up connections -----
with model:
    if hasattr(model, 'vis'):
        model.vis.setup_connections(model)
    if hasattr(model, 'ps'):
        model.ps.setup_connections(model)
    if hasattr(model, 'enc'):
        model.enc.setup_connections(model)
    if hasattr(model, 'mem'):
        model.mem.setup_connections(model)
    if hasattr(model, 'dec'):
        model.dec.setup_connections(model)
    if hasattr(model, 'mtr'):
        model.mtr.setup_connections(model)

# ----- Set up probes -----
if make_probes:
    config_and_setup_probes(model)

# ----- Neuron count debug -----
print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
if hasattr(model, 'vis'):
    print "vis n_neurons: %i" % (get_total_n_neurons(model.vis))
if hasattr(model, 'ps'):
    print "ps n_neurons: %i" % (get_total_n_neurons(model.ps))
if hasattr(model, 'bg'):
    print "bg n_neurons: %i" % (get_total_n_neurons(model.bg))
if hasattr(model, 'thal'):
    print "thal n_neurons: %i" % (get_total_n_neurons(model.thal))
if hasattr(model, 'enc'):
    print "enc n_neurons: %i" % (get_total_n_neurons(model.enc))
if hasattr(model, 'mem'):
    print "mem n_neurons: %i" % (get_total_n_neurons(model.mem))
if hasattr(model, 'dec'):
    print "dec n_neurons: %i" % (get_total_n_neurons(model.dec))
if hasattr(model, 'mtr'):
    print "mtr n_neurons: %i" % (get_total_n_neurons(model.mtr))

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
elif cfg.use_mpi:
    pass
else:
    sim = nengo.Simulator(model, dt=cfg.sim_dt)

t_build = time.time() - timestamp
timestamp = time.time()

if cfg.use_opencl or cfg.use_ref:
    print "STIMULUS SEQ: %s" % (str(stim_seq))
    print "START SIM - est_runtime: %f" % get_est_runtime()
    run_nengo_sim(sim, cfg.sim_dt, get_est_runtime(),
                  nengo_sim_run_opts=cfg.use_ref)

    t_simrun = time.time() - timestamp
    print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
    print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build, t_simrun)
else:
    print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
    print "FINISHED! - Build time: %fs" % (t_build)
    print "UPLOAD 'spaun.net' to MPI cluster to run"
    sys.exit()

if make_probes:
    # ############# WRITE PROBE DATA TO FILE ##################################
    print "WRITING PROBE DATA TO FILE"

    probe_data = {'trange': sim.trange(), 'stim_seq': stim_seq}
    for probe in sim.data.keys():
        if isinstance(probe, nengo.Probe):
            probe_data[idstr(probe)] = sim.data[probe]
    np.savez_compressed(cfg.probe_data_filename, **probe_data)

    import os
    import subprocess
    cur_dir = os.getcwd()
    subprocess.Popen(["python", os.path.join(cur_dir, 'disp_probe_data.py'),
                      cfg.probe_data_filename])
