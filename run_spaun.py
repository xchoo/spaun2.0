import os
import sys
import time
import argparse
import numpy as np

import nengo


# ----- Defaults -----
def_dim = 512
def_seq = 'A'
# def_seq = 'A0[#1]?X'
# def_seq = 'A0[#1#2#3]?XXX'
# def_seq = 'A1[#1]?XXX'
# def_seq = 'A2?XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
# def_seq = 'A3[1234]?XXXX'
def_seq = 'A3[123]?XXXX'
# def_seq = 'A3[222]?XXXX'
# def_seq = 'A3[2567589]?XXXXXXXXX'
# def_seq = 'A4[5][3]?XXXXXX'
# def_seq = 'A4[321][3]?XXXXXXX'
# def_seq = 'A4[0][9]?XXXXXXXXXXX'
# def_seq = 'A4[0][9]?XXXXXXXXXXXA3[1234321]?XXXXXXXX'
# def_seq = 'A5[123]K[3]?X'
# def_seq = 'A5[123]P[1]?X'
# def_seq = 'A6[12][2][82][2][42]?XXXXX'
# def_seq = 'A6[8812][12][8842][42][8862][62][8832]?XXXXX'
# def_seq = 'A7[1][2][3][2][3][4][3][4]?XXX'
# def_seq = 'A7[1][2][3][2]?XX'
# def_seq = 'A7[1][11][111][2][22][222][3][33]?XXXXX'
# def_seq = 'A1[1]?XXA1[22]?XX'
# def_seq = '{A1[R]?X:5}'
# def_seq = '{A3[{R:7}]?{X:8}:5}'
# def_seq = '{A3[{R:7}]?{X:8}:160}'
# def_seq = 'A3[{R:7}]?{X:8}'

def_mpi_p = 128

# ----- Definite maximum probe time (if est_sim_time > max_probe_time,
#       disable probing)
max_probe_time = 60

# ----- Add current directory to system path ---
cur_dir = os.getcwd()

# ----- Parse arguments -----
parser = argparse.ArgumentParser(description='Script for running Spaun.')
parser.add_argument(
    '-d', type=int, default=def_dim,
    help='Number of dimensions to use for the semantic pointers.')
parser.add_argument(
    '-t', type=float, default=-1,
    help=('Simulation run time in seconds. If undefined, will be estimated' +
          ' from the stimulus sequence.'))
parser.add_argument(
    '-n', type=int, default=1,
    help='Number of batches to run (each batch is a new model).')
parser.add_argument(
    '-s', type=str, default=def_seq,
    help='Stimulus sequence. Use digits to use canonical digits, prepend a ' +
         '"#" to a digit to use handwritten digits, a "[" for the open ' +
         'bracket, a "]" for the close bracket, and a "X" for each expected ' +
         'motor response. e.g. A3[1234]?XXXX or A0[#1]?X')
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
    '--seed', type=int, default=-1,
    help='Random seed to use.')
parser.add_argument(
    '--showgrph', action='store_true',
    help='Supply to show graphing of probe data.')
parser.add_argument(
    '--showanim', action='store_true',
    help='Supply to show animation of probe data.')
parser.add_argument(
    '--showiofig', action='store_true',
    help='Supply to show Spaun input/output figure.')
parser.add_argument(
    '--tag', type=str, default="",
    help='Tag string to apply to probe data file name.')
parser.add_argument(
    '--enable_cache', action='store_true',
    help='Supply to use nengo caching system when building the nengo model.')

parser.add_argument(
    '--ocl', action='store_true',
    help='Supply to use the OpenCL backend (will override -b).')
parser.add_argument(
    '--ocl_platform', type=int, default=0,
    help=('OCL Only: List index of the OpenCL platform to use. OpenCL ' +
          ' backend can be listed using "pyopencl.get_platforms()"'))
parser.add_argument(
    '--ocl_device', type=int, default=-1,
    help=('OCL Only: List index of the device on the OpenCL platform to use.' +
          ' OpenCL devices can be listed using ' +
          '"pyopencl.get_platforms()[X].get_devices()" where X is the index ' +
          'of the plaform to use.'))
parser.add_argument(
    '--ocl_profile', action='store_true',
    help='Supply to use NengoOCL profiler.')

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

parser.add_argument(
    '--nengo_gui', action='store_true',
    help='Supply to use the nengo_viz vizualizer to run Spaun.')

parser.add_argument(
    '--config', type=str, nargs='*',
    help="Use to set the various parameters in Spaun's configuration. Takes" +
         " arguments in list format. Each argument should be in the format" +
         " ARG_NAME=ARG_VALUE. " +
         "\nE.g. --config sim_dt=0.002 mb_gate_scale=0.8 " +
         "\"raw_seq_str='A1[123]?XX'\"" +
         "\nNOTE: Will override all other options that set configuration" +
         " options (i.e. --seed, --d, --s)" +
         '\nNOTE: Use quotes (") to encapsulate strings if you encounter' +
         ' problems.')

parser.add_argument(
    '--debug', action='store_true',
    help='Supply to output debug stuff.')

args = parser.parse_args()

# ----- Nengo RC Cache settings -----
# Disable cache unless seed is set (i.e. seed > 0) or if the '--enable_cache'
# option is given
if args.seed > 0 or args.enable_cache:
    print "USING CACHE"
    nengo.rc.set("decoder_cache", "enabled", "True")
else:
    print "NOT USING CACHE"
    nengo.rc.set("decoder_cache", "enabled", "False")

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

# ----- Batch runs -----
for n in range(args.n):
    print ("\n======================== RUN %i OF %i ========================" %
           (n + 1, args.n))

    # ----- Seeeeeeeed -----
    if args.seed < 0:
        seed = int(time.time())
    else:
        seed = args.seed

    cfg.set_seed(seed)
    print "MODEL SEED: %i" % cfg.seed

    # ----- Model Configurations -----
    cfg.sp_dim = args.d
    cfg.raw_seq_str = args.s
    cfg.data_dir = args.data_dir

    # Parse --config options
    if args.config is not None:
        print "USING CONFIGURATION OPTIONS: "
        for cfg_options in args.config:
            print "  * " + str(cfg_options)
            cfg_opts = cfg_options.split('=')
            cfg_param = cfg_opts[0]
            cfg_value = cfg_opts[1]
            setattr(cfg, cfg_param, eval(cfg_value))

    if cfg.use_mpi:
        sys.path.append('C:\\Users\\xchoo\\GitHub\\nengo_mpi')

        mpi_save = args.mpi_save.split('.')
        mpi_savename = '.'.join(mpi_save[:-1])
        mpi_saveext = mpi_save[-1]

        cfg.gen_probe_data_filename(mpi_savename, suffix=args.tag)
    else:
        cfg.gen_probe_data_filename(suffix=args.tag)

    make_probes = not args.noprobes

    # ----- Check if data folder exists -----
    if not(os.path.isdir(cfg.data_dir) and os.path.exists(cfg.data_dir)):
        raise RuntimeError('Data directory "%s"' % (cfg.data_dir) +
                           ' does not exist. Please ensure the correct path' +
                           ' has been specified.')

    # ----- Raw stimulus seq -----
    print "RAW STIM SEQ: %s" % (str(cfg.raw_seq_str))

    # ----- Spaun imports -----
    from _spaun.utils import get_total_n_neurons
    from _spaun.probes import idstr, config_and_setup_probes
    from _spaun.spaun_main import Spaun
    from _spaun.modules import get_est_runtime

    # ----- Spaun proper -----
    model = Spaun()

    # ----- Display stimulus seq -----
    print "PROCESSED RAW STIM SEQ: %s" % (str(cfg.raw_seq))
    print "STIMULUS SEQ: %s" % (str(cfg.stim_seq))

    # ----- Calculate runtime -----
    # Note: Moved up here so that we have data to disable probes if necessary
    runtime = args.t if args.t > 0 else get_est_runtime()

    # ----- Set up probes -----
    if runtime > max_probe_time:
        print (">>> !!! WARNING !!! EST RUNTIME > %0.2fs - DISABLING PROBES" %
               max_probe_time)
        make_probes = False

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

    if args.nengo_gui:
        print "STARTING NENGO_GUI"
        import nengo_gui
        nengo_gui.GUI(__file__, model=model, locals=locals(),
                      interactive=False).start()
        print "NENGO_GUI STOPPED"
        sys.exit()

    if cfg.use_opencl:
        import pyopencl as cl
        import nengo_ocl

        print "------ OCL ------"
        print "AVAILABLE PLATFORMS:"
        print '  ' + '\n  '.join(map(str, cl.get_platforms()))

        pltf = cl.get_platforms()[args.ocl_platform]
        print "USING PLATFORM:"
        print '  ' + str(pltf)

        print "AVAILABLE DEVICES:"
        print '  ' + '\n  '.join(map(str, pltf.get_devices()))
        if args.ocl_device >= 0:
            ctx = cl.Context([pltf.get_devices()[args.ocl_device]])
            print "USING DEVICE:"
            print '  ' + str(pltf.get_devices()[args.ocl_device])
        else:
            ctx = cl.Context(pltf.get_devices())
            print "USING DEVICES:"
            print '  ' + '\n  '.join(map(str, pltf.get_devices()))
        sim = nengo_ocl.Simulator(model, dt=cfg.sim_dt, context=ctx,
                                  profiling=args.ocl_profile)
    elif cfg.use_mpi:
        import nengo_mpi

        mpi_savefile = \
            ('+'.join([cfg.get_probe_data_filename(mpi_savename)[:-4],
                      ('%ip' % args.mpi_p if not args.mpi_p_auto else 'autop'),
                      '%0.2fs' % get_est_runtime()]) + '.' + mpi_saveext)
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
    print "BUILD FINISHED - build time: %fs" % t_build

    # ----- Spaun simulation run -----
    if cfg.use_opencl or cfg.use_ref:
        print "START SIM - est_runtime: %f" % runtime
        sim.run(runtime)

        # Close output logging file
        if hasattr(model, 'monitor'):
            model.monitor.close()

        if args.ocl_profile:
            sim.print_plans()
            sim.print_profiling()

        t_simrun = time.time() - timestamp
        print "MODEL N_NEURONS: %i" % (get_total_n_neurons(model))
        print "FINISHED! - Build time: %fs, Sim time: %fs" % (t_build,
                                                              t_simrun)
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

    # ----- Generate debug printouts -----
    n_bytes_ev = 0
    n_bytes_gain = 0
    n_bytes_bias = 0
    n_ens = 0
    for ens in sim.model.toplevel.all_ensembles:
        n_bytes_ev += sim.model.params[ens].eval_points.nbytes
        n_bytes_gain += sim.model.params[ens].gain.nbytes
        n_bytes_bias += sim.model.params[ens].bias.nbytes
        n_ens += 1

    print "## DEBUG: num bytes used for eval points: %s B" % (
        "{:,}".format(n_bytes_ev))
    print "## DEBUG: num bytes used for gains: %s B" % (
        "{:,}".format(n_bytes_gain))
    print "## DEBUG: num bytes used for biases: %s B" % (
        "{:,}".format(n_bytes_bias))
    print "## DEBUG: num ensembles: %s" % n_ens

    # ----- Close simulator -----
    if hasattr(sim, 'close'):
        sim.close()

    # ----- Write probe data to file -----
    if make_probes and not cfg.use_mpi:
        print "WRITING PROBE DATA TO FILE"

        probe_data = {'trange': sim.trange(), 'stim_seq': cfg.stim_seq}
        for probe in sim.data.keys():
            if isinstance(probe, nengo.Probe):
                probe_data[idstr(probe)] = sim.data[probe]
        np.savez_compressed(os.path.join(cfg.data_dir,
                                         cfg.probe_data_filename),
                            **probe_data)

        if args.showgrph or args.showanim:
            subprocess_call_list = ["python",
                                    os.path.join(cur_dir,
                                                 'disp_probe_data.py'),
                                    os.path.join(cfg.data_dir,
                                                 cfg.probe_data_filename),
                                    str(int(args.showgrph)),
                                    str(int(args.showanim)),
                                    str(int(args.showiofig))]

            # Log subprocess call
            if hasattr(model, 'monitor'):
                model.monitor.monitor_data.write_to_file(
                    "\n# " + " ".join(subprocess_call_list))

            print "CALLING: %s" % (" ".join(subprocess_call_list))
            import subprocess
            subprocess.Popen(subprocess_call_list)

    # ----- Write runtime data -----
    runtime_filename = os.path.join(cfg.data_dir, 'runtimes.txt')
    rt_file = open(runtime_filename, 'a')
    rt_file.write('# ---------- TIMESTAMP: %i -----------\n' % timestamp)
    rt_file.write('Backend: %s | Num neurons: %i | Tag: %s | Seed: %i\n' %
                  (cfg.backend, get_total_n_neurons(model), args.tag,
                   cfg.seed))
    if args.config is not None:
        rt_file.write('Config options: %s\n' % (str(args.config)))
    rt_file.write('Build time: %fs | Model sim time: %fs | ' % (t_build,
                                                                runtime))
    rt_file.write('Sim wall time: %fs\n' % (t_simrun))
    rt_file.close()

    # ----- Cleanup -----
    model = None
    sim = None
    probe_data = None
