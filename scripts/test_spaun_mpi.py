import numpy as np
import time
import sys

import nengo
from nengo import spa

import argparse

nengo.log(debug=False)

parser = argparse.ArgumentParser(
    description="Script for running Spaun using nengo_mpi.")

parser.add_argument(
    '-t', type=float, default=1.0,
    help='Length of the simulation in seconds.')

parser.add_argument(
    '--dir', type=str,
    default='/scratch/c/celiasmi/e2crawfo/spaun_experiments',
    help='Directory to store the probe data in.')

parser.add_argument(
    '--mpi', type=int, default=1, help='Whether to use MPI.')

parser.add_argument(
    '-p', type=int, default=1,
    help='If using MPI, the number of processors to use.')

parser.add_argument(
    '-d', type=int, default=16,
    help='The number of dimensions for the semantic pointers.')

parser.add_argument(
    '--noprog', action='store_true',
    help='Supply to omit the progress bar.')

parser.add_argument(
    '--save', default='',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

args = parser.parse_args()
print "Parameters are: ", args

probe_data_directory = args.dir

num_components = args.p

assert num_components > 0

save_file = args.save

if save_file and not args.mpi:
    raise ValueError(
        "Got %s as the filename to write to, but can't write this file "
        "because use_mpi is false.")

progress_bar = not args.noprog
setup_probes = True

# ----- Configurations -----
from _spaun.config import cfg

cfg.use_mpi = args.mpi

cfg.sp_dim = args.d
cfg.neuron_type = nengo.LIFRate()
cfg.gen_probe_data_filename("list7")

# ----- Seeeeeeeed -----
print "MODEL SEED: %i" % cfg.seed

# ----- Spaun imports -----
from _spaun.utils import get_total_n_neurons
from _spaun.vocabs import vis_vocab, pos_vocab, enum_vocab, task_vocab
from _spaun.vocabs import mtr_disp_vocab, item_vocab, vocab
from _spaun.modules import get_est_runtime
from _spaun.modules import Stimulus, Vision, ProdSys, InfoEnc, Memory, InfoDec
from _spaun.modules import Motor
from _spaun.modules.stimulus import stim_seq

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

    if hasattr(model, 'vis') and hasattr(model, 'ps'):
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
if setup_probes:
    with model:
        p0 = nengo.Probe(model.stim.output)

        pvs1 = nengo.Probe(model.vis.output, synapse=0.005)
        pvs2 = nengo.Probe(model.vis.neg_attention, synapse=0.005)
        pvs3 = nengo.Probe(model.vis.am_utilities, synapse=0.005)
        # pvs4 = nengo.Probe(model.vis.mb_output, synapse=0.005)
        # pvs5 = nengo.Probe(model.vis.vis_mb.gate, synapse=0.005)
        # pvs6 = nengo.Probe(model.vis.vis_net.output, synapse=0.005)

        pps1 = nengo.Probe(model.ps.task, synapse=0.005)
        # pps2 = nengo.Probe(model.ps.task_mb.gate, synapse=0.005)
        # pps3 = nengo.Probe(model.ps.task_mb.gateX, synapse=0.005)
        # pps4 = nengo.Probe(model.ps.task_mb.gateN, synapse=0.005)
        # pps5 = nengo.Probe(model.ps.task_mb.mem1.output, synapse=0.005)

        # pth1 = nengo.Probe(model.thal.output, synapse=0.005)

        pen1 = nengo.Probe(model.enc.pos_mb.gate, synapse=0.005)
        pen2 = nengo.Probe(model.enc.pos_mb.gateX, synapse=0.005)
        pen3 = nengo.Probe(model.enc.pos_mb.gateN, synapse=0.005)
        pen4 = nengo.Probe(model.enc.pos_mb.mem2.output, synapse=0.005)
        pen5 = nengo.Probe(model.enc.pos_mb.mem1.output, synapse=0.005)
        # pen6 = nengo.Probe(model.enc.pos_mb.am.output, synapse=0.005)
        pen6 = nengo.Probe(model.enc.pos_mb.reset, synapse=0.005)

        pmm1 = nengo.Probe(model.mem.output, synapse=0.005)
        pmm2 = nengo.Probe(model.mem.mb1a.gate, synapse=0.005)
        pmm3 = nengo.Probe(model.mem.mb1a.gateX, synapse=0.005)
        pmm4 = nengo.Probe(model.mem.mb1a.gateN, synapse=0.005)
        pmm5 = nengo.Probe(model.mem.output, synapse=0.005)
        # pmm5 = nengo.Probe(model.mem.mb1a.output, synapse=0.005)
        # pmm6 = nengo.Probe(model.mem.mb1b.output, synapse=0.005)
        # pmm7 = nengo.Probe(model.mem.mb1b.mem1.output, synapse=0.005)
        # pmm8 = nengo.Probe(model.mem.mb1b.mem2.output, synapse=0.005)
        # pmm9 = nengo.Probe(model.mem.mem_in, synapse=0.005)
        pmm6 = nengo.Probe(model.mem.mb1a.mem1.output, synapse=0.005)
        pmm7 = nengo.Probe(model.mem.mb1a.mem2.output, synapse=0.005)
        pmm8 = nengo.Probe(model.mem.mb1b.mem1.output, synapse=0.005)
        pmm9 = nengo.Probe(model.mem.mb1b.mem2.output, synapse=0.005)

        pde1 = nengo.Probe(model.dec.item_dcconv.output, synapse=0.005)
        pde2 = nengo.Probe(model.dec.select_am, synapse=0.005)
        pde3 = nengo.Probe(model.dec.select_vis, synapse=0.005)
        pde4 = nengo.Probe(model.dec.am_out, synapse=0.01)
        pde5 = nengo.Probe(model.dec.vt_out, synapse=0.005)
        pde6 = nengo.Probe(model.dec.pos_mb_gate_sig, synapse=0.005)
        pde7 = nengo.Probe(model.dec.util_diff_neg, synapse=0.005)
        pde8 = nengo.Probe(model.dec.am_utils, synapse=0.005)
        pde9 = nengo.Probe(model.dec.am2_utils, synapse=0.005)
        pde10 = nengo.Probe(model.dec.util_diff, synapse=0.005)
        pde11 = nengo.Probe(model.dec.recall_mb.output, synapse=0.005)
        pde12 = nengo.Probe(model.dec.dec_am_fr.output, synapse=0.005)
        pde13 = nengo.Probe(model.dec.dec_am.item_output, synapse=0.005)
        # pde14 = nengo.Probe(model.dec.recall_mb.mem1.output, synapse=0.005)
        pde15 = nengo.Probe(model.dec.output_know, synapse=0.005)
        pde16 = nengo.Probe(model.dec.output_unk, synapse=0.005)
        pde18 = nengo.Probe(model.dec.output_stop, synapse=0.005)
        pde19 = nengo.Probe(model.dec.am_th_utils, synapse=0.005)
        pde20 = nengo.Probe(model.dec.fr_th_utils, synapse=0.005)
        pde21 = nengo.Probe(model.dec.dec_output, synapse=0.005)
        pde22 = nengo.Probe(model.dec.dec_am_fr.input, synapse=0.005)

        pmt1 = nengo.Probe(model.mtr.ramp, synapse=0.005)
        pmt2 = nengo.Probe(model.mtr.ramp_reset_hold, synapse=0.005)
        pmt3 = nengo.Probe(model.mtr.motor_stop_input, synapse=0.005)
        pmt4 = nengo.Probe(model.mtr.motor_init, synapse=0.005)
        pmt5 = nengo.Probe(model.mtr.motor_go, synapse=0.005)

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
print "STARTING BUILD..."
print "STIMULUS SEQ: %s" % (str(stim_seq))

timestamp = time.time()

if cfg.use_mpi:
    import nengo_mpi
    assignments = {}
    partitioner = nengo_mpi.Partitioner(num_components, assignments)

    sim = nengo_mpi.Simulator(
        model, dt=cfg.sim_dt, partitioner=partitioner, save_file=save_file)
else:
    sim = nengo.Simulator(model, dt=cfg.sim_dt)

t_build = time.time() - timestamp
print "DONE BUILD"

if save_file:
    print "Saved file %s encoding the built network, exiting now." % save_file
    sys.exit()

timestamp = time.time()

sim_time = get_est_runtime()
print "START SIM - est_runtime: %f" % sim_time

# run_nengo_sim(sim, cfg.sim_dt, get_est_runtime())
sim.run(sim_time)

t_sim = time.time() - timestamp
num_neurons = get_total_n_neurons(model)

print "MODEL N_NEURONS: %i" % num_neurons
print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build, t_sim)

if setup_probes:
    print "WRITING PROBE DATA TO FILE"

    def idstr(p):
        if not isinstance(p, nengo.Probe):
            return '0'
        else:
            return str(id(p))

    version = 1.3
    probe_list = map(idstr, [p0, pvs1, pvs2, pvs3, 0,
                             p0, pps1, pmm1, pmm6, pmm7, pmm8, pmm9, 0,
                             p0, pen1, pen2, pen3, pen4, pen5, pen6, 0,
                             p0, pde1, pde2, pde3, pde4, pde5, pde6, 0,
                             p0, pmt1, pmt2, pmt3, pmt4, pmt5, 0,
                             p0, pde8, pde9, pde10, pde7, 0,
                             p0, pmm5, pde11, pde12, pde13, pde22, 0,
                             p0, pde19, pde20, pde15, pde16, pde18, 0,
                             p0, pmt1, pde21])

    vocab_dict = {idstr(pvs1): vis_vocab,
                  idstr(pps1): task_vocab,
                  idstr(pmm1): enum_vocab,
                  idstr(pmm6): enum_vocab,
                  idstr(pmm7): enum_vocab,
                  idstr(pmm8): enum_vocab,
                  idstr(pmm9): enum_vocab,
                  idstr(pmm5): item_vocab,
                  idstr(pen4): pos_vocab,
                  idstr(pen5): pos_vocab,
                  idstr(pde1): item_vocab,
                  idstr(pde4): mtr_disp_vocab,
                  idstr(pde5): mtr_disp_vocab,
                  idstr(pde11): item_vocab,
                  idstr(pde12): item_vocab,
                  idstr(pde13): item_vocab,
                  idstr(pde21): mtr_disp_vocab,
                  idstr(pde22): item_vocab}

    ############### WRITE PROBE DATA TO FILE ##################################
    probe_data = {'trange': sim.trange(), 'sp_dim': cfg.sp_dim,
                  'vocab_dict': vocab_dict, 'probe_list': probe_list,
                  'prim_vocab': vocab, 'stim_seq': stim_seq,
                  'version': version}

    for probe in sim.data.keys():
        if isinstance(probe, nengo.Probe):
            probe_data[idstr(probe)] = sim.data[probe]

    import os
    probe_data_loc = os.path.join(
        probe_data_directory, cfg.probe_data_filename)

    np.savez_compressed(probe_data_loc, **probe_data)

    # import subprocess
    # cur_dir = os.getcwd()
    # subprocess.Popen(["python", os.path.join(cur_dir, 'disp_probe_data.py'),
    #                   cfg.probe_data_filename])

    import pandas as pd

    runtimes_file = "/scratch/c/celiasmi/e2crawfo/spaun_params.csv"
    header = not os.path.isfile(runtimes_file)

    vals = vars(args).copy()
    vals.update(vars(cfg))

    vals['build_time'] = t_build
    vals['sim_time'] = t_sim

    now = pd.datetime.now()
    df = pd.DataFrame(vals, index=pd.date_range(now, periods=1))

    with open(runtimes_file, 'a') as f:
        df.to_csv(f, header=header)
