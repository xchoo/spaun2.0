
import numpy as np
import time
import sys

import nengo
from nengo import spa

nengo.log(debug=False)

import argparse

parser = argparse.ArgumentParser(
    description="Script for running Spaun using nengo_mpi.")

parser.add_argument(
    '--save', default='',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

args = parser.parse_args()
print "Parameters are: ", args

save_file = args.save

extra_partitions = 0
setup_probes = True

# ----- Configurations -----
from _spaun.config import cfg
# cfg.present_blanks = True

cfg.use_mpi = True

cfg.sp_dim = 16
cfg.neuron_type = nengo.LIFRate()
cfg.gen_probe_data_filename("list7")

print "MODEL SEED: %i" % cfg.seed

# ----- Spaun imports -----
from _spaun.utils import get_total_n_neurons
from _spaun.modules import get_est_runtime
from _spaun.modules import Stimulus
from _spaun.modules.stimulus import stim_seq

print cfg.probe_data_filename

# ----- Spaun proper -----
model = spa.SPA(label='Spaun', seed=cfg.seed)
with model:
    model.config[nengo.Ensemble].max_rates = cfg.max_rates
    model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
    model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
    model.config[nengo.Connection].synapse = cfg.pstc

    model.stim = Stimulus()

if setup_probes:
    with model:
        p0 = nengo.Probe(model.stim.output)

print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
if hasattr(model, 'vis'):
    print "vis n_neurons: %i" % (get_total_n_neurons(model.vis))

# ----- Spaun simulation run -----
print "START BUILD"
print "STIMULUS SEQ: %s" % (str(stim_seq))
timestamp = time.time()

if cfg.use_mpi:
    import nengo_mpi
    assignments = {}
    partitioner = nengo_mpi.Partitioner(1 + extra_partitions, assignments)

    sim = nengo_mpi.Simulator(
        model, dt=cfg.sim_dt, partitioner=partitioner, save_file=save_file)
else:
    sim = nengo.Simulator(model, dt=cfg.sim_dt)

if save_file:
    print "Saved file %s encoding the built network, exiting now." % save_file
    sys.exit()

t_build = time.time() - timestamp
print "BUILD TIME: %f seconds." % t_build
timestamp = time.time()

print "START SIM - est_runtime: %f" % get_est_runtime()
sim.run(get_est_runtime(), True, 'stim_output.h5')

t_simrun = time.time() - timestamp
print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build, t_simrun)