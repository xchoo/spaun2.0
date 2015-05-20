import os
import sys
import time
import argparse
import numpy as np

import nengo


# ----- Defaults -----
def_dim = 256
def_seq = 'A'
# def_seq = 'A0[1]?X'
# def_seq = 'A0[123]?XXX'
# def_seq = 'A1[1]?XXX'
# def_seq = 'A2?XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
# def_seq = 'A3[1234]?XXXX'
# def_seq = 'A3[123]?XXXX'
def_seq = 'A3[222]?XXXX'
# def_seq = 'A3[2567589]?XXXX'
# def_seq = 'A4[1][4]?XXXXXX'
# def_seq = 'A5[123]M[1]?X'
# def_seq = 'A5[123]P[1]?X'
# def_seq = 'A6[12][2][82][2][42]?X'
# def_seq = 'A7[1][2][3][2][3][4][3][4]?X'
# def_seq = 'A7[1][2][3][2]?X'

def_mpi_p = 128

# ----- Parse arguments -----
cur_dir = os.getcwd()
parser = argparse.ArgumentParser(description='Script for running Spaun.')
parser.add_argument(
    '-d', type=int, default=def_dim,
    help='Number of dimensions to use for the semantic pointers.')
parser.add_argument(
    '-t', type=float, default=-1,
    help=('Simulation run time in seconds. If undefined, will be estimated' +
          ' from the stimulus sequence.'))
parser.add_argument(
    '-s', type=str, default=def_seq,
    help='Stimulus sequence. e.g. A3[1234]?XXXX')
parser.add_argument(
    '-b', type=str, default='ref',
    help='Backend to use for Spaun. One of ["ref", "ocl", "mpi", "spinn"]')
parser.add_argument(
    '--data_dir', type=str, default=os.path.join(cur_dir, 'data'),
    help='Directory to store output data.')
parser.add_argument(
    '--noprobes', action='store_true',
    help='Supply to disable probes.')
parser.add_argument(
    '--addblanks', action='store_true',
    help=('Supply to add blanks between each character in the stimulus' +
          ' sequence.'))
parser.add_argument(
    '--present_int', type=float, default=0.15,
    help='Presentation interval of each character in the stimulus sequence.')
parser.add_argument(
    '--seed', type=int, default=-1,
    help='Random seed to use.')
parser.add_argument(
    '--showdisp', action='store_true',
    help='Supply to show graphing of probe data.')

parser.add_argument(
    '--ocl', action='store_true',
    help='Supply to use the OpenCL backend (will override -b).')
parser.add_argument(
    '--ocl_platform', type=int, default=0,
    help=('OCL Only: List index of the OpenCL platform to use. OpenCL ' +
          ' backend can be listed using "pyopencl.get_platforms()"'))

parser.add_argument(
    '--mpi', action='store_true',
    help='Supply to use the MPI backend (will override -b).')
parser.add_argument(
    '--mpi_save', type=str, default='spaun.net',
    help=('MPI Only: Filename to use to write the generated Spaun network ' +
          'to. Defaults to "spaun.net". *Note: Final filename includes ' +
          'neuron type, dimensionality, and stimulus information.'))
parser.add_argument(
    '--mpi_p', type=int, default=def_mpi_p,
    help='MPI Only: Number of processors to use.')
parser.add_argument(
    '--mpi_p_auto', action='store_true',
    help='MPI Only: Use the automatic partitioner')
parser.add_argument(
    '--mpi_compress_save', action='store_true',
    help='Supply to compress the saved net file with gzip.')

parser.add_argument(
    '--spinn', action='store_true',
    help='Supply to use the SpiNNaker backend (will override -b).')

args = parser.parse_args()

# ----- Backend Configurations -----
from _spaun.config import cfg

cfg.backend = args.b
if args.ocl:
    cfg.backend = 'ocl'
if args.mpi:
    cfg.backend = 'mpi'
if args.spinn:
    cfg.backend = 'spinn'

print "BACKEND: %s" % cfg.backend.upper()

# ----- Seeeeeeeed -----
# cfg.set_seed(1413987955)
# cfg.set_seed(1414248095)
# cfg.set_seed(1429562767)
cfg.set_seed(args.seed)
print "MODEL SEED: %i" % cfg.seed

# ----- Model Configurations -----
cfg.sp_dim = args.d
cfg.raw_seq_str = args.s
cfg.present_blanks = args.addblanks
cfg.present_interval = args.present_int
cfg.data_dir = args.data_dir

if cfg.use_mpi:
    sys.path.append('C:\\Users\\xchoo\\GitHub\\nengo_mpi')

    mpi_save = args.mpi_save.split('.')
    mpi_savename = '.'.join(mpi_save[:-1])
    mpi_saveext = mpi_save[-1]

    cfg.gen_probe_data_filename(mpi_savename)
else:
    cfg.gen_probe_data_filename()

make_probes = not args.noprobes

# ----- Spaun imports -----
from _spaun.utils import run_nengo_sim
from _spaun.utils import get_total_n_neurons
from _spaun.probes import idstr, config_and_setup_probes
from _spaun.spaun_main import Spaun
from _spaun.modules import get_est_runtime

# ----- Spaun proper -----
model = Spaun()

# ----- Set up probes -----
if make_probes:
    print "PROBE FILENAME: %s" % cfg.probe_data_filename
    config_and_setup_probes(model)

# ----- Neuron count debug -----
print "MODEL N_NEURONS:  %i" % (get_total_n_neurons(model))
if hasattr(model, 'vis'):
    print "- vis  n_neurons: %i" % (get_total_n_neurons(model.vis))
if hasattr(model, 'ps'):
    print "- ps   n_neurons: %i" % (get_total_n_neurons(model.ps))
if hasattr(model, 'bg'):
    print "- bg   n_neurons: %i" % (get_total_n_neurons(model.bg))
if hasattr(model, 'thal'):
    print "- thal n_neurons: %i" % (get_total_n_neurons(model.thal))
if hasattr(model, 'enc'):
    print "- enc  n_neurons: %i" % (get_total_n_neurons(model.enc))
if hasattr(model, 'mem'):
    print "- mem  n_neurons: %i" % (get_total_n_neurons(model.mem))
if hasattr(model, 'trfm'):
    print "- trfm n_neurons: %i" % (get_total_n_neurons(model.trfm))
if hasattr(model, 'dec'):
    print "- dec  n_neurons: %i" % (get_total_n_neurons(model.dec))
if hasattr(model, 'mtr'):
    print "- mtr  n_neurons: %i" % (get_total_n_neurons(model.mtr))

# ----- Spaun simulation build -----
print "START BUILD"
timestamp = time.time()

if cfg.use_opencl:
    import pyopencl as cl
    import nengo_ocl

    pltf = cl.get_platforms()[args.ocl_platform]
    ctx = cl.Context(pltf.get_devices())
    print "USING OPENCL - devices: %s" % (str(pltf.get_devices()))
    sim = nengo_ocl.sim_ocl.Simulator(model, dt=cfg.sim_dt, context=ctx)
elif cfg.use_mpi:
    import nengo_mpi

    mpi_savefile = ('+'.join([cfg.get_probe_data_filename(mpi_savename)[:-4],
                              ('%ip' % args.mpi_p if not args.mpi_p_auto else
                               'autop'),
                              '%0.2fs' % get_est_runtime()])
                    + '.' + mpi_saveext)
    mpi_savefile = os.path.join(cfg.data_dir, mpi_savefile)

    print "USING MPI - Saving to: %s" % (mpi_savefile)

    if args.mpi_p_auto:
        assignments = {}
        for n, module in enumerate(model.modules):
            assignments[module] = n
        sim = nengo_mpi.Simulator(model, dt=cfg.sim_dt,
                                  assignments=assignments,
                                  save_file=mpi_savefile)
    else:
        partitioner = nengo_mpi.Partitioner(args.mpi_p)
        sim = nengo_mpi.Simulator(model, dt=cfg.sim_dt,
                                  partitioner=partitioner,
                                  save_file=mpi_savefile)
else:
    sim = nengo.Simulator(model, dt=cfg.sim_dt)

t_build = time.time() - timestamp
timestamp = time.time()

# ----- Spaun simulation run -----
runtime = args.t if args.t > 0 else get_est_runtime()

if cfg.use_opencl or cfg.use_ref:
    print "STIMULUS SEQ: %s" % (str(cfg.stim_seq))
    print "START SIM - est_runtime: %f" % runtime

    if cfg.use_ref:
        run_nengo_sim(sim, cfg.sim_dt, runtime,
                      nengo_sim_run_opts=cfg.use_ref)
    else:
        sim.run(runtime)

    t_simrun = time.time() - timestamp
    print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
    print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build, t_simrun)
else:
    print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
    print "FINISHED! - Build time: %fs" % (t_build)

    if args.mpi_compress_save:
        import gzip
        print "COMPRESSING net file to '%s'" % (mpi_savefile + '.gz')

        with open(mpi_savefile, 'rb') as f_in:
            with gzip.open(mpi_savefile + '.gz', 'wb') as f_out:
                f_out.writelines(f_in)

        os.remove(mpi_savefile)

        print "UPLOAD '%s' to MPI cluster and decompress to run" % \
            (mpi_savefile + '.gz')
    else:
        print "UPLOAD '%s' to MPI cluster to run" % mpi_savefile
    t_simrun = -1

# ----- Write probe data to file -----
if make_probes and not cfg.use_mpi:
    print "WRITING PROBE DATA TO FILE"

    probe_data = {'trange': sim.trange(), 'stim_seq': cfg.stim_seq}
    for probe in sim.data.keys():
        if isinstance(probe, nengo.Probe):
            probe_data[idstr(probe)] = sim.data[probe]
    np.savez_compressed(os.path.join(cfg.data_dir, cfg.probe_data_filename),
                        **probe_data)

    if args.showdisp:
        import subprocess
        subprocess.Popen(["python", os.path.join(cur_dir,
                                                 'disp_probe_data.py'),
                          os.path.join(cfg.data_dir, cfg.probe_data_filename)])

# ----- Write runtime data -----
runtime_filename = os.path.join(cfg.data_dir, 'runtimes.txt')
rt_file = open(runtime_filename, 'a')
rt_file.write('# ---------- TIMESTAMP: %i -----------\n' % timestamp)
rt_file.write('Backend: %s | Num neurons: %i\n' %
              (cfg.backend, get_total_n_neurons(model)))
rt_file.write('Build time: %fs | Model sim time: %fs | Sim wall time: %fs\n' %
              (t_build, runtime, t_simrun))
rt_file.close()
